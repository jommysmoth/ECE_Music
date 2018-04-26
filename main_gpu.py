"""
First Attempt at Classification.

Using basic image classification to check accuracy on labels
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from CNN_GPU import Net
import time
import math
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import random
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def timesince(since):
    """
    Timing Training.

    Used to time the length of time the data
    takes to fully train
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return ' %dm %ds' % (m, s)


def labelout(output):
    """Used for accuracy testing."""
    batchsize = output.size()[0]
    batch_bal_list = []
    for examp in range(batchsize):
        samp = output[examp].data
        val, lab = samp.max(0)
        lab = lab[0]
        batch_bal_list.append(lab)
    return batch_bal_list


def print_lab(training, labels, amount):
    """Seeing how data looks."""
    for lab in labels:
        dset = training[lab]
        for x in range(amount):
            rand = np.random.randint(0, dset.shape[0])
            im = dset[rand, 0, :, :]
            plt.imshow(im)
            plt.title(lab)
            plt.figure()
    plt.show()


def ref_shape(first, second):
    """Make all images uniform."""
    rand_start = np.random.randint(first.shape[1] - second.shape[1])
    end = rand_start + second.shape[1]
    output = first[:, rand_start:end]
    return output


def rand_examp(dict_in, batches, labels):
    """
    Random Examp From Dictionary Train/Test.

    Pick from dictionary what to use.
    """
    output_list = []
    label_list = []
    for x in range(batches):
        rand_lab = np.random.randint(len(labels))
        data = dict_in[labels[rand_lab]]
        samp = random.choice(data)
        label_list.append(rand_lab)
        output_list.append(samp)
    output = np.stack(output_list, axis=0)
    output_label = np.array(label_list)
    return output, output_label


def delete_load_folder(path):
    """Used for saving space on songs for loading."""
    song_list = os.listdir(path)
    for name in song_list:
        os.remove(path + '/' + name)
    print('Old Data Erased')
    return


def main_pickle_load(which_cut, labels, batches, net_override=False):
    """Used to implement segmented data."""
    x_list = []
    y_list = []
    print('Starting train export...')
    full_train = {}
    for lab in labels:
        with open(external_file_area + '_dict/' + lab + str(which_cut) + '.pickle', 'rb') as handle:
            full_train[lab] = pickle.load(handle)
    print('Fully Loaded Data')

    for val, lab in enumerate(labels):
        for samp in full_train[lab]:
            x_list.append(samp)
            y_list.append(val)
    full_train = None
    x = np.stack(x_list, axis=0)
    print(x.shape)
    y = np.array(y_list)
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42)
    x = None
    y = None
    print('Data Split and Randomized')
    print(X_train.shape)

    h = X_train.shape[1]
    w = X_train.shape[2]
    channels = 1

    train_model_path = 'model_train/train.out'
    train_condition = not Path(train_model_path).is_file()

    if train_condition or net_override:
        cnn = Net(batches, channels, h, w, len(labels))
        cnn.cuda()
        print('Model Created')
    else:
        cnn = torch.load(train_model_path)
        cnn.cuda()
        print('Model Loaded In')

    start = time.time()

    total_train = []
    total_lab = []
    breakout = False
    train_use = X_train.shape[0]
    for x in range(train_use):
        start = int(x * batches)
        end = int(start + batches)
        if end >= train_use:
            end = int(train_use - 1)
            start = int(end - batches)
            breakout = True
        total_train.append(X_train[start:end, :, :])
        total_lab.append(y_train[start:end])
        if breakout:
            break
    X_train = None
    y_train = None
    print('Train broken into batches')
    return cnn, total_train, total_lab


if __name__ == '__main__':
    """
    ADD CROSS VALIDATION SET OVER X MANY SONGS IN TRAINING SET, MADE RANDOMLY FOR CONVERSION, THEN LOOPED OVER
    FOR LETS SAY 100 EPOOCHS AND USE CROSS VALIDATION FOR 10 PERCENT OF THE TRAINING SET AND VARY TO VALIDATE
    FOR TRAINING ONCE THE OPTIMIZATION IS DONE, SAVE MODEL AND START AGAIN FOR MORE RANDOM DATA. CHANGE TRAIN
    AMOUNT IN MAIN CLASS, SINCE TEST AND TRAIN ARE SEPERATED MORE BY FILE PATH THAN CODE. THUS, ONLY SPITS OUT
    ALL TRAINING SET AND ALL TEST SET, SPLITTING HAS TO BE DONE MANUALLY. LATER, IS SPLIT AND SEQUENTIALLY
    CHANGED FOR VALIDATION SET.
    """
    labels = ['Jazz', 'Rock', 'Rap']  # Have program output this soon
    external_file_area = 'J:/ECE_DATA/data'
    epoochs = 50
    loss_bar = range(epoochs)
    batches = 4
    learning_rate = 0.001
    cut_amount = 9
    train_model_path = 'model_train/train.out'
    loss_total = []
    loss_add = 0
    for ep in loss_bar:
        for cut in range(cut_amount):
            print(cut)
            cnn, total_train, total_lab = main_pickle_load(cut, labels, batches)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(cnn.parameters(),
                                   lr=learning_rate)
            for ind, examp in enumerate(total_train): 
                examp = torch.from_numpy(examp)
                examp = examp.type(torch.FloatTensor)
                examp = examp[:, None, :, :]

                input = Variable(examp).cuda()
                label_in_ten = torch.LongTensor(total_lab[ind])
                label_in_ten = Variable(label_in_ten).cuda()

                optimizer.zero_grad()

                output = cnn(input)
                # print(labelout(output), label_in_ten)
                loss = criterion(output, label_in_ten)
                loss.backward()
                optimizer.step()
                loss_add += loss.item()

                # print('Epooch: %i  Loss: %1.4f' % (ep, loss.data[0]))
                loss_total.append(loss.item())
            print(loss_add / ind)
            loss_add = 0
            print(loss.item())
            torch.save(cnn, train_model_path)
            print('Model Saved')
            total_train = None
            total_lab = None
    plt.plot(loss_total)
    plt.show()
