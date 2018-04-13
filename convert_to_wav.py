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


class ConvertToWav:
    """
    Converting Files.

    Changes files into wav format for use in python, also adding label for
    non-folder use function
    """

    def __init__(self, seconds_clip, folder_method=None):
        """
        Constants.

        Will list expected form under variable
        """
        self.folder_method = folder_method
        self.seconds = seconds_clip
        # String, or NoneType, depending on type needed.

    def mp3_to_wav(self, pathname):
        """
        MP3 To WAV converter.

        Now have it working so that if only mp3 files exist, it changes format
        of all mp3 go to wav in a different folder. When files are mixed (wav, flac,
        mp3, etc), code will be updated for transers. For now, this is functioning as
        spectrogram proof of concept
        """
        song_strings = glob.glob(pathname)
        for song in song_strings:
            mp3_form = AudioSegment.from_mp3(song)
            song_name = song[5:-4]
            mp3_form.export('data_wav/' + song_name + '.wav',
                            format='wav')
        """
        This is the portion of the code used for when the data is sorted in folders,
        instead of relying on metadata. Default is None, giving the above functional
        use.

        Gives added Function of label file creation
        """
        if self.folder_method == 'folder':
            label_list = open('data_wav/labels.txt', 'w')
            ab_path_cwd = os.getcwd()
            for path, direc, files in os.walk(ab_path_cwd + pathname):
                for file in files:
                    filetype = file[-4:]
                    filename = file[:-4]
                    print(filename)
                    if filetype == '.mp3':
                        label_name = path.replace(ab_path_cwd + pathname, "")
                        mp3_form = AudioSegment.from_mp3(path + '/' + file)
                        half_point = int(len(mp3_form) / 2)
                        full_clip = self.seconds * 1000  # Runs in miliseconds
                        mp3_wav = mp3_form[half_point:(half_point + full_clip)]
                        mp3_wav.export('data_wav/' + filename + '.wav',
                                       format='wav')
                        label_list.write('%s : %s\n' % (filename, label_name))
                    else:
                        if file[-5:] == '.flac':
                            filetype = file[-5:]
                            filename = file[:-5]
                            label_name = path.replace(ab_path_cwd + pathname, "")
                            data, samprate = sf.read(path + '/' + file)
                            half_point = int(data.shape[0] / 2)
                            # set value due to different sample sizes
                            set_value = self.seconds * 44100
                            new_data = data[half_point:int(half_point + set_value)]
                            sf.write('data_wav/' + filename + '.wav', new_data, samprate)
                            label_list.write('%s : %s\n' % (filename, label_name))
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


if __name__ == '__main__':

    # pathname = "/data/*.mp3"
    pathname = '/data/'
    ctw = ConvertToWav(3, 'folder')
    ctw.mp3_to_wav(pathname)
    final_path = 'data_wav'
    # make_label_file(pathname, final_path)
