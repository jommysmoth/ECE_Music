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
import itertools
import math
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import random
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
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


def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
    """
    Function prints and plots the confusion matrix.

    Taken from scipy for ease.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def shuffle_in_unison(a, b):
    """Used to shuffle data."""
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def main_pickle_load(which_cut, labels, batches, net_override=False, train=True):
    """
    Used to load in chunk method for training / testing.

    Might need to make simpler at some point
    """
    x_list = []
    y_list = []
    full = {}
    if train:
        add = '_train'
    else:
        add = '_test'
    for lab in labels:
        with open(external_file_area + '_dict/' + lab + add + str(which_cut) + '.pickle', 'rb') as handle:
            full[lab] = pickle.load(handle)
    print('Loaded Chunk %i' % which_cut)
    for val, lab in enumerate(labels):
        for samp in full[lab]:
            x_list.append(samp)
            y_list.append(val)
    x = np.stack(x_list, axis=0)
    y = np.array(y_list)
    if train:
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42)
        # Not touching validation right quick
        print('Data Split and Randomized')
    else:
        x_train, y_train = shuffle_in_unison(x, y)
        print('Data Randomized')
    x = None
    y = None

    h = x_train.shape[1]
    w = x_train.shape[2]
    channels = 1

    train_model_path = '../model_train/train.out'
    train_condition = not Path(train_model_path).is_file()
    if train_condition or net_override:
        cnn = Net(batches, channels, h, w, len(labels))
    else:
        cnn = torch.load(train_model_path)
        print('Model Loaded In')
    total_train = []
    total_lab = []
    breakout = False
    train_use = x_train.shape[0]
    for x in range(train_use):
        start = int(x * batches)
        end = int(start + batches)
        if end >= train_use:
            end = int(train_use - 1)
            start = int(end - batches)
            breakout = True
        total_train.append(x_train[start:end, :, :])
        total_lab.append(y_train[start:end])
        if breakout:
            break
    x_train = None
    y_train = None
    print('Broken into batches')
    return cnn, total_train, total_lab, x_val, y_val


def val_find(model, x_val, y_val):
    """Output Val accuracy."""
    model.eval()
    total = 0
    correct = 0
    for run in range(x_val.shape[0]):
        examp = torch.from_numpy(x_val[run, :, :])
        examp = examp.type(torch.FloatTensor)
        examp = examp[None, None, :, :]
        images = Variable(examp)
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += 1
        if predicted[0] == y_val[run]:
            correct += 1
    accuracy = correct / total
    return accuracy

if __name__ == '__main__':
    """
    Train / Test is 80 / 20 Right Now
    """
    labels = ['Jazz', 'Rock', 'Rap', 'Folk', 'Classical', 'Electronic']  # Have program output this soon
    override_convert = False
    update_songs = 5000
    net_override = False
    override_process = False
    train_samples = None
    plot_results = True
    more_training = False
    external_file_area = '../data'
    procd = cst.ProcessingData(labels, seconds_total=30,
                               data_folder=external_file_area,
                               override_convert=override_convert,
                               conversions=update_songs,
                               ext_storage=external_file_area)
    dict_dest = external_file_area + '_dict/' + random.choice(labels)
    condition_train = not Path(dict_dest + '_train1.pickle').is_file()
    condition_test = not Path(dict_dest + '_test.pickle').is_file()
    if condition_train or condition_test or override_process:
        for lab in labels:
            if not override_process:
                train, test = procd.main_train_test(lab, condition_train, condition_test)
                if condition_train:
                    with open(external_file_area + '_dict/' + lab + '_train.pickle', 'wb') as handle:
                        pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if condition_test:
                    with open(external_file_area + '_dict/' + lab + '_test.pickle', 'wb') as handle:
                        pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                train, test = procd.main_train_test(lab, train_spec=True, test_spec=True)
                with open(external_file_area + '_dict/' + lab + '_train.pickle', 'wb') as handle:
                    pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(external_file_area + '_dict/' + lab + '_test.pickle', 'wb') as handle:
                    pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Saved ' + lab)
        if override_process:
            print('Overwrote Train / Test Data')
        else:
            print('Train / Test Data Saved')
        exit()
    else:
        print('Data Loading')

    epoochs = 10
    batches = 30
    channels = 1
    learning_rate = 0.001
    start = time.time()
    chunk_used = 1
    train_model_start = '../model_train/train_' + 'drpout_.2_'
    train_model_path = train_model_start + str(epoochs) + '_epoochs_chunk_' + str(chunk_used) + '.out'
    train_condition = not Path(train_model_path).is_file()

    if train_condition or more_training:
        cnn, total_train, total_lab, x_val, y_val = main_pickle_load(chunk_used, labels, batches)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(cnn.parameters(),
                              lr=learning_rate,
                              momentum=0.9)
        loss_graph = []
        loss_bar = tqdm(range(epoochs))
        for ep in loss_bar:
            loss_sum = 0
            for ind, examp in enumerate(total_train):
                examp = torch.from_numpy(examp)
                examp = examp.type(torch.FloatTensor)
                examp = examp[:, None, :, :]

                input = Variable(examp)
                label_in_ten = torch.LongTensor(total_lab[ind])
                label_in_ten = Variable(label_in_ten)

                optimizer.zero_grad()

                output = cnn(input)
                # print(output, label_in_ten)
                loss = criterion(output, label_in_ten)
                loss.backward()
                optimizer.step()
                loss_sum += loss.data[0]
                loss_graph.append(loss.data[0])
                # val_accuracy = val_find(cnn, x_val, y_val)
                # print(val_accuracy)
            loss_bar.set_description('Epooch: %i  Loss: %1.4f' % (ep, loss_sum / ind))
            torch.save(cnn, train_model_path)
        print('Validation Accuracy of: %1.4f' % val_find(cnn, x_val, y_val))
        plt.plot(loss_graph)
        plt.show()
        exit()
    else:
        full_test = {}
        for lab in labels:
            with open(external_file_area + '_dict/' + lab + '_test.pickle', 'rb') as handle:
                full_test[lab] = pickle.load(handle)
        cnn = torch.load(train_model_path)
        print('Load Test Data and Trained Model')
    cnn.eval()
    _, total_train, total_lab, x_val, y_val = main_pickle_load(chunk_used, labels, batches)
    print('Validation Accuracy of: %1.4f' % val_find(cnn, x_val, y_val))
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    for val, lab in enumerate(labels):
        for ind, examp in enumerate(full_test[lab]):
            examp = torch.from_numpy(examp)
            examp = examp.type(torch.FloatTensor)
            examp = examp[None, None, :, :]
            label_ten = torch.LongTensor([val])
            images = Variable(examp)
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.append(predicted[0])
            y_true.append(val)
            total += 1
            correct += (predicted == label_ten).sum()
    print('Test Accuracy is : %1.4f' % (correct / total))
    train_pred = []
    train_true = []
    for run, examp in enumerate(total_train):
        for bat in range(batches):
            examp_ = torch.from_numpy(examp[bat, :, :])
            examp_ = examp_.type(torch.FloatTensor)
            examp_ = examp_[None, None, :, :]
            images = Variable(examp_)
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            train_pred.append(predicted[0])
            train_true.append(total_lab[run][bat])
    if plot_results:
        cm_train = confusion_matrix(train_true, train_pred)
        train_title = 'Train: Epoochs =  %i , Learning Rate = %1.4f' % (epoochs, learning_rate)
        test_title = 'Test: Epoochs =  %i , Learning Rate = %1.4f' % (epoochs, learning_rate)
        plot_confusion_matrix(cm_train, classes=labels, title=train_title)
        plt.figure()
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, classes=labels, title=test_title)
        plt.show()
