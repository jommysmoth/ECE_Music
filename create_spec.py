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
import matplotlib.pyplot as plt
from pathlib import Path
from convert_to_wav import ConvertToWav
import librosa
import pyaudio
import wave
from tqdm import tqdm


class ProcessingData:
    """
    ProcessingData Training.

    Used for creating the training set of processed images,
    """

    def __init__(self, labels, seconds_total, data_folder, override_convert,
                 conversions, ext_storage, test_split=0.2):
        """
        Instance.

        Used for Label list, length of song clips, and more as needed
        """
        self.labels = labels
        self.seconds_total = seconds_total
        self.path = data_folder
        self.override = override_convert
        self.conversions = conversions
        self.ext_storage = ext_storage
        self.test_split = test_split

    def find_label(self, label, test=False):
        """
        Adding label to data component.

        Solid relative path now, with be made modular if necessary
        """
        label_list = []
        if test:
            start_file = self.path + '_wav_test/'
        else:
            start_file = self.path + '_wav/'
        labels_doc = open(start_file + 'labels.txt', 'r')
        end_file = '.wav'
        with labels_doc as doc:
            data = doc.readlines()
            for row in data:
                name, label_val = row.split(' : ')
                if label_val == label + '\n':
                    label_list.append(start_file + name + end_file)
        return label_list

    def listen_to_clip(self, song):
        """Checking for correct labels."""
        chunk = 1024

        f = wave.open(song, "rb")

        p = pyaudio.PyAudio()

        stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                        channels=f.getnchannels(),
                        rate=f.getframerate(),
                        output=True)
        data = f.readframes(chunk)

        while data:
            stream.write(data)
            data = f.readframes(chunk)

        stream.stop_stream()
        stream.close()

        p.terminate()
        return

    def add_label(self, label_save, test=False):
        """
        Adding label to data component.

        Holder for now, skim metadata later
        """
        label = label_save
        song_strings = self.find_label(label, test)
        song_list = []
        for song in tqdm(song_strings):
            # self.listen_to_clip(song)
            y, sr = librosa.load(song, mono=True)
            sp = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=323,
                                                n_fft=4096, hop_length=2048)
            sp = librosa.power_to_db(sp, ref=np.max)
            sp /= np.mean(sp)
            song_list.append(sp)
        # Each list entry has dim of Set Amount Per Label- Data_X - Data_Y - Channels
        if test:
            print(label + ' Conversion For Testing Done')
        else:
            print(label + ' Conversion For Training Done')
        return song_list

    def left_right_mix(self, song, samp_rate):
        """
        Mixing All Channels.

        Using the channels to create a whole scaled image
        """
        chn = []
        channel_amount = song.shape[1]
        for lr in range(channel_amount):
            sp, freqs, bins, im = plt.specgram(song[:, 0], Fs=samp_rate)
            sg = im.get_array()
            print(sg.shape)
            sg = sg / np.mean(sg)
            chn.append(sg)
        left_right_stacked = np.stack(chn, axis=0)
        return left_right_stacked  # Channel - Data_X - Data_Y

    def main_train_test(self, label_save, train_spec=True, test_spec=False):
        """
        Split Data.

        Split data set into train and test sets.
        """
        train_dict = {}
        test_dict = {}
        pathname = self.path
        ctw = ConvertToWav(self.seconds_total, pathname, self.conversions, self.ext_storage)
        ctw_test = ConvertToWav(self.seconds_total, pathname,
                                self.conversions * self.test_split, self.ext_storage, test=True)
        train_cond = not Path(pathname + '_wav/' + 'Data_Loaded.txt').is_file()
        test_cond = not Path(pathname + '_wav_test/' + 'Data_Loaded.txt').is_file()
        if train_cond:
            ctw.mp3_to_wav(pathname)
            open(pathname + '_wav/' + 'Data_Loaded.txt', 'w')
            print('Converted Training Data')
        if test_cond:
            ctw_test.mp3_to_wav(pathname)
            open(pathname + '_wav_test/' + 'Data_Loaded.txt', 'w')
            print('Converted Testing Data')
        if self.override:
            ctw.mp3_to_wav(pathname)
            ctw_test.mp3_to_wav(pathname)
            print('Data Written Over')
        if (not train_cond) and (not test_cond) and (not self.override):
            print('Data Already Converted')
        if train_spec:
            train_dict = self.add_label(label_save)
        else:
            train_dict = {}
        if test_spec:
            test_dict = self.add_label(label_save,
                                       test=True)
        else:
            test_dict = {}
        return train_dict, test_dict
