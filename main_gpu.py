"""
First Attempt at Classification.

Using basic image classification to check accuracy on labels
"""
import create_spec as cst
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from CNN import Net
import time
import math
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import random
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split


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
    override_convert = False
    update_songs = 15000
    net_override = True
    override_process = False
    train_samples = None
    external_file_area = '/media/jommysmoth/Storage/ECE_DATA/data'
    procd = cst.ProcessingData(labels, train_amount=0.7,
                               seconds_total=30,
                               data_folder=external_file_area,
                               override_convert=override_convert,
                               conversions=update_songs,
                               ext_storage=external_file_area,
                               train_samples=train_samples)
    condition = not Path('data_dict/train.pickle').is_file() and not Path('data_dict/test.pickle').is_file()

    if condition or override_process:
        for lab in ['Rap']:
            train, test = procd.main_train_test(lab)
            with open(external_file_area + '_dict/' + lab + '.pickle', 'wb') as handle:
                pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
            """
            with open('data_dict/test.pickle', 'wb') as handle:
                pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
            """
            print('Saved ' + lab)
        if override_process:
            print('Overwrote Train / Test Data')
        else:
            print('Train / Test Data Saved')
        exit()
    else:
        full_train = {}
        for lab in labels:
            with open(external_file_area + '_dict/' + lab + '.pickle', 'rb') as handle:
                full_train[lab] = pickle.load(handle)
    # delete_load_folder('data_wav')

    epoochs = 50
    batches = 32
    X_list = []
    y_list = []
    for val, lab in enumerate(labels):
        for samp in full_train[lab]:
            X_list.append(samp)
            y_list.append(val)
    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    h = X_train.shape[1]
    w = X_train.shape[2]
    channels = 1
    learning_rate = 0.001

    train_model_path = 'model_train/train.out'
    train_condition = not Path(train_model_path).is_file()
    if train_condition or net_override:
        cnn = Net(batches, channels, h, w, len(labels))
        cnn.cuda()
    else:
        cnn = torch.load(train_model_path)
        cnn.cuda()
        print('Model Loaded In')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(),
                           lr=learning_rate)
    start = time.time()
    loss_bar = range(epoochs)
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

    for ep in loss_bar:
        for ind, examp in enumerate(total_train):
            examp = torch.from_numpy(examp)
            examp = examp.type(torch.FloatTensor)
            examp = examp[:, None, :, :]

            input = Variable(examp).cuda()
            label_in_ten = torch.LongTensor(total_lab[ind])
            label_in_ten = Variable(label_in_ten).cuda()

            optimizer.zero_grad()

            output = cnn(input)
            # print(output, label_in_ten)
            loss = criterion(output, label_in_ten)
            loss.backward()
            optimizer.step()
            print('Epooch: %i  Loss: %1.4f' % (ep, loss.data[0]))
