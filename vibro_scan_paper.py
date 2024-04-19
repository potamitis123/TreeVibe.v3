
# -*- coding: utf-8 -*-
"""
Code for paper: Automatic vibroacoustic monitoring of trees against borers

You need the 'infested' folder and the 'silence' folder to get the thresholds
https://zenodo.org/records/10820310
@author: potam
"""

from pydub import AudioSegment
from pydub.utils import mediainfo

import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def convert_rec_to_float(data):
    if data.dtype == np.uint8:
        data = (data - 128) / 128.
    elif data.dtype == np.int16:
        data = data / 32768.
    elif data.dtype == np.int32:
        data = data / 2147483648.
    return data

def plot_figure(audio, vad, file_path, counts, multi_level_thres):

    t = np.arange(len(audio)) / 16000

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    # Code for the top subplot
    ax1.plot(t, audio, label='Vibrational recording')
    ax1.plot(t, .85*vad, label='Detection location', color='red', linewidth=3)
    ax1.set_ylabel('Amplitude')
    ax1.set_ylim(-1, 1)
    ax1.legend()
    ax1.set_title(file_path[-23:-4] + ' Det:' + str(counts))
    
    # Code for the second subplot
    energy_values, chunk_energy = calculate_energy(audio)
    ax2.plot(t, energy_values, label='Energy of the recording')
    ax2.plot(t, multi_level_thres*chunk_energy, label='Energy of the thresholds', linestyle='dotted', color='red', linewidth=3)
    ax2.set_xlabel('time (sec)')
    ax2.set_ylabel('Amplitude')
    ax2.legend()

    #plt.savefig('D:/Trees/TO_UPLOAD/test_set/mullbery_7_days/automatic_pics/'+file_path[-23:-4]+'.png', dpi=300, bbox_inches='tight')
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


def get_silence_thres(mp3_silence):
    
    total_energy_values = np.zeros(1)
    for filename in os.listdir(mp3_silence):
        if filename.endswith(".mp3"):
            file_path = os.path.join(mp3_silence, filename)

            # Process the MP3 file
            # info = mediainfo(file_path)
            # fs = info['sample_rate']
            audio = np.array(AudioSegment.from_mp3(file_path).get_array_of_samples())
            audio = convert_rec_to_float(audio)

            # Calculate the energy of each sample
            energy_values, _ = calculate_energy(audio)

            # Ensure total_energy_values has the same shape as energy_values
            if total_energy_values.shape[0] != energy_values.shape[0]:
                total_energy_values = np.zeros_like(energy_values, dtype=float)            
            total_energy_values = np.add(total_energy_values, energy_values)
            
    return total_energy_values/len(os.listdir(mp3_silence))   

def process_mp3_file(file_path, min_duration_threshold, max_duration_threshold, total_energy_values, rec_thres, multi_level_thres, reject_noisy):
    # Load the MP3 file
    audio = np.array(AudioSegment.from_mp3(file_path).get_array_of_samples())
    #audio = audio[1500:480000]
    audio = convert_rec_to_float(audio)

    vad = np.zeros_like(audio)
    
    # Calculate the energy of each sample
    energy_values, chunk_energy = calculate_energy(audio)

    # Set the energy threshold
    threshold_energy = multi_level_thres*chunk_energy; # 10 is a strict threshold. To process the high quality infested folder set thres=3 and remove file rejection
    comparison_value = 500*np.mean(total_energy_values)
    perm_chunk = [1 if value < comparison_value else 0 for value in chunk_energy]

    chunks = []
    chunk_start = None
    durations = []

    n_energy_values = np.sum(energy_values)/len(energy_values)
    n_total_energy_values = np.sum(total_energy_values)/len(total_energy_values)
    energy_ratio = n_energy_values/n_total_energy_values

    if reject_noisy and energy_ratio > rec_thres:
        #print(f"{file_path[-23:]}: {np.sum(energy_values)/np.sum(total_energy_values)}")
        #plot_figure(audio, vad, file_path, 0, multi_level_thres)
        return audio, [], [], 0, energy_ratio
  
    for i, energy_sample in enumerate(energy_values):
        #if energy_sample >= threshold_energy[i] and perm_chunk[i]:
        if energy_sample >= threshold_energy[i]:
            
            if chunk_start is None:
                # Start of a new chunk
                chunk_start = i
        elif chunk_start is not None:
            # End of a chunk
            chunk_duration = i - chunk_start
            if chunk_duration < max_duration_threshold and chunk_duration > min_duration_threshold:

               #with open('D:/Trees/TO_UPLOAD/output.txt', 'a') as output_file:
                   #output_file.write(f"{file_path[-23:]}: start: {chunk_start} end: {chunk_start+chunk_duration} duration: {chunk_duration} chunk_energy: {np.sum(energy_values[chunk_start:chunk_start + chunk_duration]):.1f}\n")
               durations.append(chunk_duration)
               vad[chunk_start:chunk_start+chunk_duration] = 1
            chunks.append((chunk_start, chunk_duration))
            chunk_start = None

    # Check for the last chunk
    if chunk_start is not None:
        chunk_duration = len(energy_values) - chunk_start
        chunks.append((chunk_start, chunk_duration))
        if chunk_duration < max_duration_threshold and chunk_duration > min_duration_threshold:

            #with open('D:/Trees/TO_UPLOAD/output.txt', 'a') as output_file:
                #output_file.write(f"{file_path[-23:]}: start: {chunk_start} end: {chunk_start+chunk_duration} duration: {chunk_duration} chunk_energy: {np.sum(energy_values[chunk_start:chunk_start + chunk_duration]):.1f}\n")
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
    # Infested folders
    mp3_directory = "D:/Trees/infested/"
    #mp3_directory = "D:/Trees/TO_UPLOAD/test_set/infested_1"
    #mp3_directory = "D:/Trees/TO_UPLOAD/test_set/infested_2"
    #mp3_directory = "D:/Trees/TO_UPLOAD/test_set/infested_3"
    #mp3_directory = "D:/Trees/TO_UPLOAD/test_set/infested_4"
    #mp3_directory = "D:/Trees/TO_UPLOAD/test_set/infested_5"

    # Not infested folders
    #mp3_directory = "D:/Trees/not_infested/"
    #mp3_directory = "D:/Trees/not_infested_tree/" # Fig tree
    #mp3_directory = "D:/Trees/not_infested_fig/" # Fig tree

    # Get global energy thresholds from silence
    mp3_silence = "D:/Trees/silence/"
    total_energy_values = get_silence_thres(mp3_silence)            
    
    # Specify the threshold duration and energy threshold
    min_duration_threshold = 150 #127 #for in-lab experiments
    max_duration_threshold = 350 #650 #for in-lab experiments #
    rec_thres = 185
    multi_level_thres = 7 #3 #for in-lab experiments
    reject_noisy = False # When scanning field data this needs to be set to True. The 'infested' folder has mostly high quality data

    total_durations = []
    chunks_per_file = []
    cumulative_counter = []
    total_counter = 0
    filenames = []
    dur_energy = []
    index = 1
    accepted = []
    
    # Iterate over each MP3 file in the directory
    for filename in np.sort(os.listdir(mp3_directory)):
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

    df.to_csv("D:/Trees/stereo/woodboring_impulses_annotation2.csv", index=False)

    # (1) Create a figure with two subplots
    from matplotlib.ticker import MaxNLocator
    
    x = range(1,len(cumulative_counter)+1)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Plot on the first subplot
    ax1.plot(x, cumulative_counter, label='Cumulative sum of impulses')
    ax1.set_ylabel('Counts')
    #ax1.grid(True)
    ax1.legend()
    
    # Plot on the second subplot
    ax2.plot(x, chunks_per_file, label='Number of impulses per file')
    ax2.set_xlabel('# File id')
    ax2.set_ylabel('Counts')
    #ax2.grid(True)
    ax2.legend()
    
    # Adjust layout to prevent clipping of labels
    plt.tight_layout()
    
    # Show the plot
    plt.show()


    # (2) Filename and no. of impulses per file are only needed
    df = pd.DataFrame({'filenames': filenames, 'counts': chunks_per_file})
    df['timestamp'] = pd.to_datetime(df['filenames'].str.slice(2, 16), format='%Y%m%d%H%M%S')
    
    # Convert timestamp to UTC
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    
    # Correct timestamp to +2 hours
    df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Athens')  # Change 'Europe/Berlin' to your specific time zone
    df['hour'] = df['timestamp'].dt.hour
    
    # Aggregate counts by hour
    hourly_counts = df.groupby('hour')['counts'].sum().reset_index()
    
    # Circular barplot
    theta = np.linspace(0.0, 2 * np.pi, len(hourly_counts), endpoint=False)
    width = (2*np.pi) / len(hourly_counts)
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.bar(theta, hourly_counts['counts'], width=width, align="center", alpha=0.7)
    
    # Beautify the plot
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_yticks([])
    
    # Add labels
    labels = [f'{hour:02d}' for hour in hourly_counts['hour']]
    ax.set_xticks(theta)
    ax.set_xticklabels(labels)
    ax.set_title(r'Activity of $\mathit{X.chinensis}$ around the clock')

    # Display the circular barplot
    plt.show()

    
    # (3) Create a figure for durations
    flattened_list = flatten_list(total_durations)
    #plt.hist(flattened_list,100);plt.grid(True, linestyle='--', alpha=0.7);plt.xlabel('samples');plt.ylabel('Counts');plt.title('Impulse duration')
    q_msec=[i*1000/16000 for i in flattened_list if i<=400] # keep only up to 25 msec for better visualisation
    plt.hist(q_msec,100);plt.grid(True, linestyle='--', alpha=0.7);plt.xlabel('msec');plt.ylabel('# Impulses');plt.title('Impulse duration')