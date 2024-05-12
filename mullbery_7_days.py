# -*- coding: utf-8 -*-
"""
Code for paper: Automatic vibroacoustic monitoring of trees against borers
You need /mullbery_7_days folder from the test set and infested_tree_alley.xlsx that holds the annotation
@author: potam
"""

from pydub import AudioSegment

import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

def plot_figure(audio, vad, file_path, counts, multi_level_thres):

    t = np.arange(len(audio)) / 16000

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    # Code for the top subplot
    ax1.plot(t, audio/32768, label='Vibrational recording')
    ax1.plot(t, .8*vad, label='Detection location', color='red', linewidth=3)
    ax1.set_ylabel('Amplitude')
    ax1.set_ylim(-1, 1)
    ax1.legend()
    ax1.set_title(file_path[-23:-4] + ' Det:' + str(counts))
    
    # Code for the second subplot
    energy_values, chunk_energy = calculate_energy(audio/32768)
    ax2.plot(t, energy_values, label='Energy of the recording')
    ax2.plot(t, multi_level_thres*chunk_energy, label='Energy of the thresholds', linestyle='dotted', color='red', linewidth=3)
    ax2.set_xlabel('time (sec)')
    ax2.set_ylabel('Amplitude')
    ax2.legend()

    #plt.savefig('D:/Trees/TO_UPLOAD/test_set/mullbery_7_days/automatic_pics/'+file_path[-23:-4]+'.png')
    plt.show() # comment out this line for batch processing
    plt.close(fig)
    return


def calculate_energy(audio_segment):
    # Calculate energy as the squared value of each sample
    #samples = audio_segment.get_array_of_samples()
    samples = audio_segment
    energy = np.array([sample**2 for sample in samples])

    # Apply a convolution kernel to smooth the energy values
    M = 200 # smooth events
    smoothed_energy = np.convolve(energy, np.ones(M)/M, mode='same')
    
    N = 10; # Number of chunks
    chunk_size = len(smoothed_energy) // N; # Size of each chunk
    
    # Initialize variables
    chunk_energy = np.zeros(len(smoothed_energy))

    # Partition the recording and calculate energy for each chunk
    for i in range(N):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(smoothed_energy))
    
        # Extract the current chunk
        current_chunk = smoothed_energy[start_idx:end_idx]
    
        # Calculate energy for the current chunk (you can customize the energy calculation method)
        chunk_energy[start_idx:end_idx] = np.sum(current_chunk) / len(current_chunk)

    return smoothed_energy, chunk_energy

def get_sil_thres(mp3_silence):
    total_energy_values = np.zeros(1)
    for filename in os.listdir(mp3_silence):
        if filename.endswith(".mp3"):
            file_path = os.path.join(mp3_silence, filename)

            audio = AudioSegment.from_mp3(file_path).get_array_of_samples()
            audio = audio[:480000]

            # Calculate the energy of each sample
            energy_values, _ = calculate_energy(audio)

            total_energy_values = np.add(total_energy_values, energy_values)
            
    return total_energy_values/len(os.listdir(mp3_silence))  

def process_mp3_file(file_path, min_duration_threshold, max_duration_threshold, total_energy_values, rec_thres, multi_level_thres, reject_noisy):
    # Load the MP3 file
    audio = np.array(AudioSegment.from_mp3(file_path).get_array_of_samples())  # Mono
    #audio = audio[1500:]

    vad = np.zeros_like(audio)
    
    # Calculate the energy of each sample
    energy_values, chunk_energy = calculate_energy(audio)

    # Set the energy threshold
    threshold_energy = multi_level_thres*chunk_energy; # 10 is a strict threshold. To process the high quality infested folder set thres=3 and remove file rejection

    chunks = []
    chunk_start = None
    durations = []

    energy_ratio = np.sum(energy_values)/np.sum(total_energy_values)

    if reject_noisy and energy_ratio > rec_thres:
        #print(f"{file_path[-23:]}: {energy_ratio}")
        #plot_figure(audio, vad, file_path, 0, multi_level_thres)
        return audio, [], [], 0, energy_ratio
  
    for i, energy_sample in enumerate(energy_values):
        if energy_sample >= threshold_energy[i]:
            
            if chunk_start is None:
                # Start of a new chunk
                chunk_start = i
        elif chunk_start is not None:
            # End of a chunk
            chunk_duration = i - chunk_start
            if chunk_duration < max_duration_threshold and chunk_duration > min_duration_threshold:

               durations.append(chunk_duration)
               vad[chunk_start:chunk_start+chunk_duration] = 1
            chunks.append((chunk_start, chunk_duration))
            chunk_start = None

    # Check for the last chunk
    if chunk_start is not None:
        chunk_duration = len(energy_values) - chunk_start
        chunks.append((chunk_start, chunk_duration))
        if chunk_duration < max_duration_threshold and chunk_duration > min_duration_threshold:

            durations.append(chunk_duration)
            vad[chunk_start:chunk_start+chunk_duration] = 1

    # Shortlist chunks shorter than the specified duration threshold. We have derived segment length from experimentation
    shortlisted_chunks = [(start, np.sum(energy_values[start:start + duration])) for start, duration in chunks if duration < max_duration_threshold and duration > min_duration_threshold]
    
    #plot_figure(audio, vad, file_path, len(shortlisted_chunks), multi_level_thres)
    return audio, shortlisted_chunks, durations, 1, energy_ratio

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in (flatten_list(sublist) if isinstance(sublist, list) else [sublist])]

def main():
    # Specify the directory containing the MP3 files
    mp3_directory = "D:/Trees/TO_UPLOAD/test_set/mullbery_7_days"

    # Get global energy thresholds from silence
    mp3_silence = "D:/Trees/silence/"
    
    total_energy_values = get_sil_thres(mp3_silence)            
    
    # Specify the threshold duration and energy threshold
    min_duration_threshold = 150 
    max_duration_threshold = 350 
    rec_thres = 10
    multi_level_thres = 7 
    reject_noisy = True

    total_durations = []
    chunks_per_file = []
    cumulative_counter = []
    total_counter = 0
    filenames = []
    dur_energy = []
    index = 1
    accepted = []
    
    true_column = pd.read_excel('D:/Trees/TO_UPLOAD/test_set/infested_tree_alley.xlsx', usecols=['filenames','TRUTH'])
    
    # Iterate over each MP3 file in the directory
    for filename in true_column.filenames:
        if filename.endswith(".mp3") or filename.endswith(".wav"):
            file_path = os.path.join(mp3_directory, filename)

            # Process the MP3 file
            audio, shortlisted_chunks, file_durations, reject_flag, p = process_mp3_file(file_path, min_duration_threshold, max_duration_threshold, total_energy_values, rec_thres, multi_level_thres, reject_noisy)

            total_durations.append(file_durations)
            shs = [impulse_energy for start, impulse_energy in shortlisted_chunks]
            dur_energy.append(shs)
            # Print or save the results as needed
            print(f"#{index} Pulses in {filename}: {len(shortlisted_chunks)}")
            accepted.append(reject_flag)
            filenames.append(filename)
            chunks_per_file.append(len(shortlisted_chunks))
            total_counter += len(shortlisted_chunks)
            cumulative_counter.append(total_counter)
            index+=1

    df = pd.DataFrame({'filenames': filenames, 'counts': chunks_per_file, 'accept': accepted})

    # Regression metrics
    
    print(f"RMSE: {np.sqrt(mean_squared_error(true_column.TRUTH, df.counts))}")
    print(f"MAE: {mean_absolute_error(true_column.TRUTH, df.counts)}")
    print(f"CORR: {true_column.TRUTH.corr(df.counts)}")
    print(f"CORR log: {np.log(true_column.TRUTH+1).corr(np.log(df.counts+1))}")
    print(f"(%) accepted recs: {sum(accepted)/len(accepted)}")

    
    # Binary classification metrics

    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    # Derive stats for accepted recs

    e = [[true_column.TRUTH[i], df.counts[i]] for i in range(len(df)) if accepted[i] == 1]
    df1 = pd.DataFrame({0:e})
    df1[['Truth', 'Pred']] = df1[0].apply(pd.Series)
    df1.drop(columns=[0], inplace=True)
    y_test = df1.Truth>0
    y_pred = df1.Pred>0

    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    print(classification_report(y_test, y_pred))

    # cumulative sum for accepted recordings
    extracted_values1 = [true_column.TRUTH[i] for i in range(len(cumulative_counter)) if accepted[i] == 1]
    extracted_values2 = [chunks_per_file[i] for i in range(len(chunks_per_file)) if accepted[i] == 1]
    t = np.arange(len(extracted_values1))
    plt.plot(t, np.cumsum(extracted_values1), label='Ground Truth', linewidth=3)
    plt.plot(t, np.cumsum(extracted_values2), label='Automatic counting', linewidth=3)
    
    # Add labels and legend
    plt.xlabel('# Files')
    plt.ylabel('Cumulative sum')
    plt.title('Automatic classification of infestation status')
    plt.legend()


    
