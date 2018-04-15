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


if __name__ == '__main__':
    # labels = ['Alternative', 'Experimental Rock', 'Grindcore', 'Hardcore', 'Indie Rock', 'Post Rock']
    labels = ['Rock', 'Rap']
    procd = cst.ProcessingData(labels, train_amount=0.7)
    data = procd.main()
    print_amount = 10
    size = 0
    for lab in labels:
        size += data[lab].shape[0]
    # print_lab(data, labels, 3)

    data_array = np.zeros((size, data[labels[0]].shape[1],
                           data[labels[0]].shape[2],
                           data[labels[0]].shape[3]))
    start = 0
    data_labels = []
    for ind, lab in enumerate(labels):
        end = start + data[lab].shape[0]
        data_array[start:end, :, :, :] = data[lab]
        start = data[lab].shape[0] - 1
        for x in range(data[lab].shape[0]):
            data_labels.append(ind)

    data_labels = np.array(data_labels)
    name = 'Data_Set %s' % (data_array.shape,)
    np.savetxt(name, data_array.flatten())
    np.savetxt('Label_Set.out', data_labels, delimiter=',')

    exit()

    X_train, X_test, y_train, y_test = train_test_split(data_array, data_labels,
                                                        test_size=0.01,
                                                        random_state=40)

    runs = 500
    batches = 30
    channels = X_train.shape[1]
    h = X_train.shape[2]
    w = X_train.shape[3]
    print_every = 3
    learning_rate = 0.001

    cnn = Net(batches, channels, h, w, len(labels))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(),
                           lr=0.005)

    start = time.time()
    train_size = X_train.shape[0]
    exit()
    for run in range(runs):
        total_loss = 0
        count = 0
        for ins in range(X_train.shape[0]):
            data_start = (ins * batches)
            data_end = data_start + batches - 1
            if data_end >= train_size:
                data_end = train_size
            label_in = y_train[data_start:data_end]
            # print(data_start, data_end, train_size)
            label_in = [int(x) for x in label_in]

            examp = X_train[data_start:data_end, :, :, :]

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
            # print(loss.data[0])

            guess = labelout(output)
            total_loss += loss.data[0]
            count += 1
            if data_end == train_size:
                break
        print(guess, label_in)
        print(total_loss / count)
        plt.plot(run, total_loss / count)
        print(timesince(start))
    plt.show()
