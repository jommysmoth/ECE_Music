"""
Data Analysis.

Going through random .wav files, and trying to understand features
"""
import random
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    plt.inferno()
    labels = ['Jazz', 'Rock', 'Rap']
    external_file_area = '/media/jommysmoth/Storage/ECE_DATA/data'
    label_list = []
    random_label = 'Jazz'
    start_file = external_file_area + '_wav/'
    labels_doc = open(start_file + 'labels.txt', 'r')
    end_file = '.wav'
    with labels_doc as doc:
            data = doc.readlines()
            for row in data:
                name, label_val = row.split(' : ')
                if label_val == random_label + '\n':
                    label_list.append(start_file + name + end_file)
    random_song = label_list[1500]
    y, sr = librosa.load(random_song, mono=True)
    sp = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=323,
                                        n_fft=4096, hop_length=2048)
    sp = librosa.power_to_db(sp, ref=np.max)
    sp /= np.mean(sp)
    examp = torch.from_numpy(sp)
    examp = examp.type(torch.FloatTensor)
    examp = examp[None, None, :, :]
    examp = Variable(examp)
    channels = 1
    conv1_shape = (323, 1)
    conv1_shape = 1
    conv2_shape = 5
    pool_1_shape = 2
    pool_2_shape = 2
    conv1_outshape = 5
    conv2_outshape = 10
    seq = nn.Sequential(nn.Conv2d(in_channels=channels,
                                  out_channels=conv1_outshape,
                                  kernel_size=conv1_shape,
                                  stride=1,
                                  padding=0),
                        nn.BatchNorm2d(conv1_outshape),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=pool_1_shape))
    seq2 = nn.Sequential(nn.Conv2d(in_channels=conv1_outshape,
                                             out_channels=conv2_outshape,
                                             kernel_size=conv2_shape,
                                             stride=5,
                                             padding=2),
                                   nn.BatchNorm2d(conv2_outshape),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=pool_2_shape))
    output = seq(examp)
    output = seq2(output)
    np_output = output.data[0].numpy()
    print(np_output.shape)
    for x in range(np_output.shape[0]):
        plt.figure()
        plt.imshow(np_output[x, :, :])
    plt.show()
