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

CROSSED OUT
Right now, only set up to deal with equal label amount of data, if this becomes issue could make function
output a dictionary with different label numpy arrays instead of full numpy result.
CROSSED OUT

Work on things next:
- Create library for learning (Drag and drop song selections to folder of genre label)
- - Try to use similar sample rates, since it is important for gaining information of same level
- Start using logistic regression / other basic learning techniques
- Start convolutional development
- Turn function modules to classes (maybe if important evantually)
- Change needed form in classifier UNTIL final classifier

"""


import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt


class ProcessingData:
    """
    ProcessingData Training.

    Used for creating the training set of processed images,
    """

    def __init__(self, labels, train_amount, clip_type=10000):
        """
        Instance.

        Used for Label list, length of song clips, and more as needed
        """
        self.labels = labels

        if isinstance(clip_type, int):
            self.clip = clip_type
        elif isinstance(clip_type, str):
            if isinstance(int(clip_type), int):
                self.clip = clip_type
        else:
            print('Need different clipping method/amount input')
            raise KeyError('Dumby')
        self.tr_split = train_amount

    def clipping_song(self, song, seconds_clip):
        """
        File Clipping.

        For now uses random 1/3 to 1/2 clipping
        """
        samp_rate, song_array = wav.read(song)
        sig_len = song_array.shape[0]
        rand_start = np.random.randint(sig_len / 3, sig_len / 2)
        if isinstance(self.clip, int):
            rand_end = rand_start + self.clip
        else:
            # Start with simple 44100 sample-rate
            rand_end = rand_start + (int(self.clip) * 44100)
        clipped_song = song_array[rand_start:rand_end, :]
        return clipped_song, samp_rate

    def find_label(self, label):
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

    def add_label(self, labels, seconds_clip):
        """
        Adding label to data component.

        Holder for now, skim metadata later
        """
        label_dict = {}
        label_list = []
        for label in labels:
            song_strings = self.find_label(label)
            song_list = []
            for song in song_strings:
                song_cl_array, samp_rate = self.clipping_song(song, seconds_clip)
                song_lr = self.left_right_mix(song_cl_array, samp_rate)
                song_list.append(song_lr)
            label_list.append(np.stack(song_list))
            # Each list entry has dim of Set Amount Per Label- Data_X - Data_Y - Channels
        for ind, dict_fill in enumerate(label_list):
            label_dict[labels[ind]] = dict_fill
        return label_dict

    def left_right_mix(self, song, samp_rate):
        """
        Mixing All Channels.

        Using the channels to create a whole scaled image
        """
        chn = []
        channel_amount = song.shape[1]
        for lr in range(channel_amount):
            spectrum, freq, time, im = plt.specgram(song[:, lr],
                                                    Fs=44100)

            chn.append(spectrum)
        left_right_stacked = np.stack(chn, axis=2)
        return left_right_stacked  # Data_X - Data_Y - Channel

    def main(self):
        """
        Output Function.

        Used to retrieve numpy array necessary for training.
        """
        # song_strings = glob.glob('data_wav/*.wav')  # only necessary for non-folder input
        labels = self.labels
        seconds_clip = 3
        data_dict = self.add_label(labels,
                                   seconds_clip)
        # Label Dictionary:Set Amount Per Label-Data_X-Data_Y-Channels
        return data_dict, labels

    def rand_train_test_main(self):
        """
        Output Function.

        Used to retrieve numpy array necessary for training.
        """
        # song_strings = glob.glob('data_wav/*.wav')  # only necessary for non-folder input
        labels = self.labels
        seconds_clip = 3
        data_dict = self.add_label(labels,
                                   seconds_clip)
        # Label Dictionary:Set Amount Per Label-Data_X-Data_Y-Channels
        train = {}
        test = {}
        for lab in labels:
            dset = data_dict[lab]
            lab_size = dset.shape[0]
            train_r = int(self.tr_split * lab_size)
            shuf = np.arange(lab_size)
            np.random.shuffle(shuf)
            train_list = shuf[:train_r]
            test_list = shuf[:-train_r]
            train_array = np.delete(dset, test_list, 0)
            test_array = np.delete(dset, train_list, 0)
            train[lab] = train_array
            test[lab] = test_array
        return train, test
