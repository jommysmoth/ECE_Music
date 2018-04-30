"""
Used for exploring different elements of models.

Weight vectors, how images are vizualized throughout network, etc...
"""
import torch.nn as nn
import torch
from CNN import Net
import os
import random
from pydub import AudioSegment
import librosa
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


def fake_network(image, output_size):
    """Used to Visualize Network."""
    height = image.shape[3]
    list_along_inside = []
    conv1_shape = (height, 1)
    conv2_shape = 5
    pool_1_shape = 2
    conv1_outshape = 20
    conv2_outshape = 40
    conv1 = nn.Conv2d(in_channels=1,
                      out_channels=conv1_outshape,
                      kernel_size=conv1_shape,
                      stride=1,
                      padding=2)
    relu = nn.ReLU()
    maxpool = nn.MaxPool2d(kernel_size=pool_1_shape)
    conv2 = nn.Conv2d(in_channels=conv1_outshape,
                      out_channels=conv2_outshape,
                      kernel_size=conv2_shape,
                      stride=1,
                      padding=2)
    batch_norm = nn.BatchNorm2d(conv2_outshape)
    dense_1 = nn.Linear(40 * 81, 1024)
    dense_mid = nn.Linear(1024, 256)
    dense_out = nn.Linear(256, output_size)

    inner = conv1(image)
    list_along_inside.append(inner)
    inner = relu(inner)
    list_along_inside.append(inner)
    inner = maxpool(inner)
    list_along_inside.append(inner)
    inner = conv2(inner)
    list_along_inside.append(inner)
    inner = batch_norm(inner)
    list_along_inside.append(inner)
    inner = relu(inner)
    list_along_inside.append(inner)
    inner = maxpool(inner)
    list_along_inside.append(inner)
    inner = inner.view(inner.size(0), -1)
    inner = dense_1(inner)
    list_along_inside.append(inner)
    inner = dense_mid(inner)
    inner = dense_out(inner)
    return list_along_inside


if __name__ == '__main__':
    labels = ['Jazz', 'Rock', 'Rap', 'Folk', 'Classical', 'Electronic']
    model_path = '../model_train/'
    external_example = '/media/jommysmoth/Storage/ECE_DATA/example'
    example_label = 'Rock'
    model_list = os.listdir(model_path)
    example_name = os.listdir(external_example)
    title = ['First Convolutional Layer',
             'First Convolutional Layer (With RelU activation)',
             'First Max Pooling Layer',
             'Second Convolutional Layer',
             'Second Convolutional Layer (With ReLU activation)',
             'Second Convolutional Layer (With ReLU and Batch Normalization)',
             'Second MaxPool2d Layer',
             'First Fully Connected Layer']
    if len(example_name) == 1:
        mp3_form = AudioSegment.from_mp3(external_example + '/' + example_name[0])
        full_clip = 30 * 1000  # Runs in miliseconds
        rand_start = np.random.randint(0, len(mp3_form) - full_clip)
        mp3_wav = mp3_form[rand_start:(rand_start + full_clip)]
        destin = external_example + '/' + example_name[0] + '.wav'
        mp3_wav.export(destin, format='wav')
    song = external_example + '/' + example_name[1]
    y, sr = librosa.load(song, mono=True)
    sp = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=323,
                                        n_fft=4096, hop_length=2048)
    sp = librosa.power_to_db(sp, ref=np.max)
    sp /= np.mean(sp)
    plt.inferno()
    plt.imshow(sp)
    examp = torch.from_numpy(sp)
    examp = examp.type(torch.FloatTensor)
    examp = examp[None, None, :, :]
    image = Variable(examp)
    list_viz = fake_network(image, len(labels))
    for ind, viz_im in enumerate(list_viz):
        im = viz_im.data.numpy()
        if len(im.shape) > 2:
            im_show = np.transpose(im[0, np.random.randint(0, im.shape[0]), :, :])
        else:
            im_show = im[0, :]
        plt.figure()
        plt.plot(im_show, 'o')
        plt.title(title[ind])
    for model in model_list:
        cnn = torch.load(model_path + model)
        cnn.eval()
        output = cnn(image)
        _, predicted = torch.max(output.data, 1)
        string = model + ' predicted label ' + labels[predicted[0]]
        string2 = '. The correct label is ' + example_label
        print(string + string2)
        print('\n\n\n')
    plt.show()
