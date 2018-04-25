"""
MP3 to Wav Conversion.

For files I have that are not already in a WAV format, which
is naive to use in python, this file will convert to a useable
format

Might label music from folders containing genre name, depending on amount of meta data availible for music.
Could be a form of "labeling by hand", since sub-genre is less labeled. Worth doing function for both
instances, since could become a necessity.

Changed Clipping to this to save on data storage
"""
from pydub import AudioSegment
import soundfile as sf
import glob
from mutagen.id3 import ID3
import os
import random
from tqdm import tqdm
import numpy as np
from pathlib import Path
import tempfile


class ConvertToWav:
    """
    Converting Files.

    Changes files into wav format for use in python, also adding label for
    non-folder use function
    """

    def __init__(self, seconds_clip, path, conversions, ext_storage, folder_method='folder'):
        """
        Constants.

        Will list expected form under variable
        """
        self.folder_method = folder_method
        self.seconds = seconds_clip
        self.path = path
        self.out_folder = path + '_wav'
        self.total_conv = conversions
        self.ext_storage = ext_storage
        # String, or NoneType, depending on type needed.

    def assign_new_name(self, destination_old):
        """
        Assign Temp Song name (only need genre and name to data munch).

        Make it so files with same name (but ultimately different portions of the song)
        can be saved independently
        """
        tf = tempfile.NamedTemporaryFile(prefix='song')
        filename = tf.name[5:]
        destination = self.out_folder + '/' + filename + '.wav'
        if destination == destination_old:
            filename, destination = self.assign_new_name(destination)
        return filename, destination

    def mp3_to_wav(self, pathname):
        """
        MP3 To WAV converter.

        Now have it working so that if only mp3 files exist, it changes format
        of all mp3 go to wav in a different folder. When files are mixed (wav, flac,
        mp3, etc), code will be updated for transers. For now, this is functioning as
        spectrogram proof of concept

        song_strings = glob.glob(pathname)
        for song in song_strings:
            mp3_form = AudioSegment.from_mp3(song)
            song_name = song[5:-4]
            mp3_form.export('data_wav/' + song_name + '.wav',
                            format='wav')

        This is the portion of the code used for when the data is sorted in folders,
        instead of relying on metadata. Default is None, giving the above functional
        use.

        Gives added Function of label file creation
        """
        if self.folder_method == 'folder':
            label_list = open(self.out_folder + '/' + 'labels.txt', 'w')
            my_path_cwd = self.ext_storage
            labels = os.listdir(my_path_cwd)
            pbar = tqdm(range(self.total_conv))
            for upl in pbar:
                rand_lab = random.choice(labels)
                path = my_path_cwd + '/' + rand_lab
                file = random.choice(os.listdir(path))
                filetype = file[-4:]
                filename = file[:-4]
                pbar.set_description(filename)
                if filetype == '.mp3':
                    mp3_form = AudioSegment.from_mp3(path + '/' + file)
                    full_clip = self.seconds * 1000  # Runs in miliseconds
                    if full_clip > len(mp3_form):
                        continue
                    rand_start = np.random.randint(0, len(mp3_form) - full_clip)
                    mp3_wav = mp3_form[rand_start:(rand_start + full_clip)]
                    destin = self.out_folder + '/' + filename + '.wav'
                    if Path(destin).is_file():
                        filename, destin = self.assign_new_name(destin)
                    mp3_wav.export(destin, format='wav')
                    label_list.write('%s : %s\n' % (filename, rand_lab))
                else:
                    if file[-5:] == '.flac':
                        filetype = file[-5:]
                        filename = file[:-5]
                        data, samprate = sf.read(path + '/' + file)
                        # set value due to different sample sizes
                        set_value = self.seconds * samprate
                        if set_value > len(data):
                            continue
                        rand_start = np.random.randint(0, len(data) - set_value)
                        new_data = data[rand_start:int(rand_start + set_value)]
                        destin = self.out_folder + '/' + filename + '.wav'
                        if Path(destin).is_file():
                            filename, destin = self.assign_new_name(destin)
                        sf.write(destin, new_data, samprate)
                        label_list.write('%s : %s\n' % (filename, rand_lab))
        return

    def make_label_file(self, pathname, final_path):
        """
        Make label file.

        Give pathname as data_wav or final folder, depending on extent of data crunching used
        in the end result
        """
        song_strings = glob.glob(pathname)
        genre_list = []
        for song in song_strings:
            audio = ID3(song)
            boy = 'TCON'
            if boy in list(audio.keys()):
                genre_list.append([str(audio['TIT2']), str(audio[boy])])

        label_list = open('data_wav/labels.txt', 'w')
        for item in genre_list:
            label_list.write('%s : %s\n' % (item[0], item[1]))
