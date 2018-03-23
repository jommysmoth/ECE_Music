"""
Creating Spectrogram.

This program is for taking in the wav files created in order to create the image representation
of the music needed for the ml algorithm. This may end up, evantually, including the feature embedding
for song length or other features deemed necessary. Also, file clipping might occur here, depending on how
extensive the coding necessary is (or if a serperate module would be useful for clipping).

IMPORTANT
Figuring out where best to clip from, start with random sample in 1/3 to 1/2 of the way through a song,
for 5 second, (SEE IF SAMP RATE AFFECTS SPEC IMAGE SINCE MORE DATA POINTS IN SAME TIME, MIGHT START WITH
1000 DATA POINTS, OR BASELINE 44.1KHZ ASSUMPTION FOR 5 SECONDS)
"""


import numpy as np
import scipy.io.wavfile as wav
import glob
import matplotlib.pyplot as plt


def clipping_song(song, seconds_clip):
    """
    File Clipping.

    For now uses random 1/3 to 1/2 clipping
    """
    samp_rate, song_array = wav.read(song)
    sig_len = song_array.shape[0]
    rand_start = np.random.randint(sig_len / 3, sig_len / 2)
    rand_end = rand_start + (seconds_clip * samp_rate)
    clipped_song = song_array[rand_start:rand_end, :]
    return clipped_song, samp_rate


def find_label(label):
    """
    Adding label to data component.

    Solid relative path now, with be made modular if necessary
    """
    label_list = []
    labels_doc = open('data_wav/labels.txt', 'r')
    start_file = 'data_wav/'
    end_file = '.wav'
    with labels_doc as doc:
        data = doc.readlines()
        for row in data:
            name, label_val = row.split(' : ')
            if label_val == label + '\n':
                label_list.append(start_file + name + end_file)
    return label_list


def add_label(labels, seconds_clip):
    """
    Adding label to data component.

    Holder for now, skim metadata later
    """
    label_list = []
    for label in labels:
        song_strings = find_label(label)
        song_list = []
        for song in song_strings:
            song_cl_array, samp_rate = clipping_song(song, seconds_clip)
            song_lr = left_right_mix(song_cl_array, samp_rate)
            song_list.append(song_lr)
        label_list.append(np.stack(song_list))
        # Each list entry has dim of Set Amount Per Label- Data_X - Data_Y - Channels
    full_array = np.stack(label_list)  # Label - Set Amount Per Label- Data_X - Data_Y - Channels
    return full_array


def left_right_mix(song, samp_rate):
    """
    Mixing All Channels.

    Using the channels to create a whole scaled image
    """
    chn = []
    channel_amount = song.shape[1]
    for lr in range(channel_amount):
        spectrum, freq, time, im = plt.specgram(song[:, lr],
                                                Fs=samp_rate,
                                                NFFT=int(samp_rate * 0.005),
                                                noverlap=int(samp_rate * 0.0025))
        chn.append(spectrum)
    left_right_stacked = np.stack(chn, axis=2)
    return left_right_stacked  # Data_X - Data_Y - Channel


if __name__ == '__main__':
    song_strings = glob.glob('data_wav/*.wav')
    labels = ['Test_Label_1', 'Test_Label_2']  # Made by hand, since honestly not many labels
    seconds_clip = 3
    data_array = add_label(labels, seconds_clip)  # Label - Set Amount Per Label- Data_X - Data_Y - Channels
    print(data_array.shape)
