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


def norm(input):
    """Used to normalize by channel."""
    for chn in range(input.shape[1]):
        cur = input[:, chn, :, :]
        input[:, chn, :, :] = cur / np.mean(cur)
    return input


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


def shuffle(a, b):
    """CHANGE."""
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def run_data(data_start, data_end, train_data, train_labels, cnn, optimizer, criterion):
    """
    Training.

    Pushes data through network
    """
    label_in = train_labels[data_start:data_end]
    print(data_start, data_end, train_size)
    label_in = [int(x) for x in label_in]

    examp = train_data[data_start:data_end, :, :, :]

    examp = torch.from_numpy(examp)
    examp = examp.type(torch.FloatTensor)

    input = Variable(examp)
    label_in_ten = torch.LongTensor(label_in)
    label_in_ten = Variable(label_in_ten)

    optimizer.zero_grad()

    output = cnn(input)
    loss = criterion(output, label_in_ten)
    loss.backward()
    optimizer.step()

    guess = labelout(output)

    print(len(guess))
    print(len(label_in))
    return guess, label_in, loss


if __name__ == '__main__':
    labels = ['Alternative', 'Experimental Rock', 'Grindcore', 'Hardcore', 'Indie Rock', 'Post Rock']
    procd = cst.ProcessingData(labels, train_amount=0.7)
    train, test, train_size, test_size = procd.rand_train_test_main()
    print_amount = 10
    train_labels = []
    test_labels = []

    train_data = np.zeros((train_size, train['Grindcore'].shape[1],
                           train['Grindcore'].shape[2],
                           train['Grindcore'].shape[3]))
    last_en = 0
    for ind, lab in enumerate(labels):
        start = last_en
        end = (last_en + train[lab].shape[0])
        train_data[last_en:(last_en + train[lab].shape[0]), :, :, :] = train[lab]
        last_en += train[lab].shape[0]
        for am in range(train[lab].shape[0]):
            train_labels.append(ind)

    train_labels = np.array(train_labels, dtype=int)
    train_data, train_labels = shuffle(train_data, train_labels)

    runs = 50
    batches = 30
    channels = train_data.shape[1]
    h = train_data.shape[2]
    w = train_data.shape[3]
    print_every = 3

    cnn = Net(batches, channels, h, w, len(labels))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(),
                          lr=0.005,
                          momentum=0.9)

    start = time.time()
    print(train_data.shape[0])
    for run in range(runs):
        for ins in range(train_data.shape[0]):
            data_start = (ins * batches)
            data_end = data_start + batches - 1
            if data_end >= train_size:
                label_in = train_labels[data_start:data_end]
                print(data_start, data_end, train_size)
                label_in = [int(x) for x in label_in]

                examp = train_data[data_start:data_end, :, :, :]

                examp = torch.from_numpy(examp)
                examp = examp.type(torch.FloatTensor)

                input = Variable(examp)
                label_in_ten = torch.LongTensor(label_in)
                label_in_ten = Variable(label_in_ten)

                optimizer.zero_grad()

                output = cnn(input)
                loss = criterion(output, label_in_ten)
                loss.backward()
                optimizer.step()

                guess = labelout(output)
                break
            label_in = train_labels[data_start:data_end]
            print(data_start, data_end, train_size)
            label_in = [int(x) for x in label_in]

            examp = train_data[data_start:data_end, :, :, :]

            examp = torch.from_numpy(examp)
            examp = examp.type(torch.FloatTensor)

            input = Variable(examp)
            label_in_ten = torch.LongTensor(label_in)
            label_in_ten = Variable(label_in_ten)

            optimizer.zero_grad()

            output = cnn(input)
            loss = criterion(output, label_in_ten)
            loss.backward()
            optimizer.step()

            guess = labelout(output)
        print(guess, label_in)
        print(loss.data[0])
        print(timesince(start))
