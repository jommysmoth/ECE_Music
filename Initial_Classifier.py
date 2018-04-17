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


def scan_spec(input, w):
    """
    Scanning through spec data.

    Assuming a format for input of

    # of Songs (should be one before input, giving 3d input)
    Channels
    h
    amount of song seconds * w
    Should keep consistent label input for random scans of songs
    """
    output = np.empty(input.shape[0], input.shape[1], w)
    upper = input.shape[1] - w
    rand_val = np.random.randint(0, upper)
    rand_upper = rand_val + w
    output = input[:, :, rand_val:rand_upper]
    return output


def rand_examp(dict_in, batches, labels, w):
    """
    Random Examp From Dictionary Train/Test.

    Pick from dictionary what to use.
    """
    output_list = []
    label_list = []
    for x in range(batches):
        rand_lab = np.random.randint(2)
        data = dict_in[labels[rand_lab]]
        samp = data[np.random.randint(data.shape[0]), :, :, :]
        output_list.append(scan_spec(samp, w))
        label_list.append(rand_lab)
    output = np.stack(output_list, axis=0)
    output_label = np.array(label_list)
    return output, output_label


if __name__ == '__main__':
    # labels = ['Alternative', 'Experimental Rock', 'Grindcore', 'Hardcore', 'Indie Rock', 'Post Rock']
    labels = ['Rock', 'Rap']
    override_convert = False
    procd = cst.ProcessingData(labels, train_amount=0.7,
                               seconds_total=30,
                               data_folder='data',
                               override_convert=override_convert)
    train, test = procd.main()
    override = True

    n_iter = 10000
    batches = 30
    start_example, not_needed = rand_examp(train, 2, labels, 100)  # just needed for height
    channels = start_example.shape[1]
    h = start_example.shape[2]
    w = h
    learning_rate = 0.001
    stop = 100

    cnn = Net(batches, channels, h, w, len(labels))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(),
                           lr=learning_rate)

    start = time.time()
    total_loss = 0
    count = 0

    for run in range(n_iter):
        examp, label_in = rand_examp(train, w)

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
        total_loss += loss.data[0]
        count += 1
        if run % stop == 0:
            guess = labelout(output)
            print(guess, label_in)
            print(total_loss / count)
            plt.plot(run, total_loss / count)
            print(timesince(start))
            total_loss = 0
            count = 0
    plt.show()
