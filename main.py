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


def rand_examp(dict_in, batches, labels, force):
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
        if samp.shape[1] > force:
            rand_start = np.random.randint(samp.shape[1] - force)
            end = rand_start + force
            samp = samp[:, rand_start:end]
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
    labels = ['Jazz', 'Rock', 'Rap']  # Have program output this soon
    not_done = True
    override_convert = True
    update_songs = 100
    clean_after_pickle = True
    net_override = False
    override_process = True
    external_file_area = '/media/jommysmoth/Storage/ECE_DATA/data'
    procd = cst.ProcessingData(labels, train_amount=0.7,
                               seconds_total=30,
                               data_folder='data',
                               override_convert=override_convert,
                               conversions=update_songs,
                               ext_storage=external_file_area)
    condition = not Path('data_dict/train.pickle').is_file() and not Path('data_dict/test.pickle').is_file()
    while not_done:
        if condition or override_process:
            train, test = procd.main_train_test()
            with open('data_dict/train.pickle', 'wb') as handle:
                pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('data_dict/test.pickle', 'wb') as handle:
                pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if override_process:
                print('Overwrote Train / Test Data')
            else:
                print('Train / Test Data Saved')
        else:
            with open('data_dict/train.pickle', 'rb') as handle:
                train = pickle.load(handle)
            with open('data_dict/test.pickle', 'rb') as handle:
                test = pickle.load(handle)
            print('Train / Test Data Loaded')
        delete_load_folder('data_wav')
        n_iter = 1000
        batches = 30
        force = 10000
        start_example, not_needed = rand_examp(train, batches, labels, force)  # just needed for height
        h = start_example.shape[1]
        w = start_example.shape[2]
        channels = 1
        learning_rate = 0.001
        stop_show = n_iter / 10
        stop_plot = n_iter / 50

        train_model_path = 'model_train/train.out'
        train_condition = not Path(train_model_path).is_file()
        if train_condition or net_override:
            cnn = Net(batches, channels, h, w, len(labels))
        else:
            cnn = torch.load(train_model_path)
            print('Model Loaded In')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(cnn.parameters(),
                              lr=learning_rate,
                              momentum=0.9)
        start = time.time()
        total_loss = 0
        print_loss = 0
        count = 0
        count_plot = 0
        test_tot = 10
        all_losses = []
        loss_bar = tqdm(range(n_iter))

        for run in loss_bar:
            examp, label_in = rand_examp(train, batches, labels, force)

            examp = torch.from_numpy(examp)
            examp = examp.type(torch.FloatTensor)
            examp = examp[:, None, :, :]

            input = Variable(examp)
            label_in_ten = torch.LongTensor(label_in)
            label_in_ten = Variable(label_in_ten)

            optimizer.zero_grad()

            output = cnn(input)
            # print(output, label_in_ten)
            loss = criterion(output, label_in_ten)
            loss.backward()
            optimizer.step()
            # print(loss.data[0])
            total_loss += loss.data[0]
            count += 1
            count_plot += 1
            print_loss += loss.data[0]
            guess = labelout(output)
            accuracy = sum(1 for x, y in zip(guess, label_in) if x == y) / len(guess)
            if run % stop_show == 0:
                loss_bar.set_description('Loss: %1.4f, Accuracy: %0.4f' % (print_loss / count, accuracy))
                print_loss = 0
                count = 0
            if run % stop_plot == 0:
                all_losses.append(total_loss / count_plot)
                total_loss = 0
                count_plot = 0
        torch.save(cnn, train_model_path)
        print('Model Saved')
        net_override = False
        if loss.data[0] < 0.2:
            not_done = False
            plt.plot(all_losses)
            plt.show()
        """
        suc = 0
        attempts = 0
        for test_am in range(test_tot):
            rand_test, true_label = rand_examp(test, batches, labels, force)
            rand_test = torch.from_numpy(rand_test)
            rand_test = rand_test.type(torch.FloatTensor)
            rand_test = rand_test[:, None, :, :]
            rand_test = Variable(rand_test)
            output = cnn(rand_test)
            guess = labelout(output)
            for ind, g in enumerate(guess):
                if g == true_label[ind]:
                    suc += 1
                attempts += 1
        accuracy = (suc / attempts) * 100
        print('\n\n Accuracy of the model is: %f' % (accuracy))
        plt.plot(all_losses)
        plt.show()
        """
