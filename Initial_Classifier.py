"""
First Attempt at Classification.

Using basic image classification to check accuracy on labels
"""
import create_spec as cst
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from CNN import Net

if __name__ == '__main__':
    labels = ['Experimental Rock', 'Grindcore', 'Hardcore', 'Indie Rock', 'Post Rock']
    procd = cst.ProcessingData(labels, train_amount=0.8)
    train, test, train_size, test_size = procd.rand_train_test_main()
    # Label Dictionary:Set Amount Per Label-Data_X-Data_Y-Channels
    train_labels = []
    test_labels = []

    train_data = np.zeros((train_size, train['Grindcore'].shape[1],
                           train['Grindcore'].shape[2],
                           train['Grindcore'].shape[3]))
    last_en = 0
    for ind, lab in enumerate(labels):
        train_data[last_en:(last_en + train[lab].shape[0]), :, :, :] = train[lab]
        last_en = train[lab].shape[0]
        for am in range(train[lab].shape[0]):
            train_labels.append(ind)

    train_labels = np.array(train_labels, dtype=int)

    runs = 2
    batches = 4
    channels = train_data.shape[1]
    h = train_data.shape[2]
    w = train_data.shape[3]
    print(channels, h, w)

    cnn = Net(batches, channels, h, w, len(labels))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(),
                          lr=0.001,
                          momentum=0.9)

    for run in range(runs):
        for ins in range(train_data.shape[0]):
            examp = train_data[ins:(batches + ins), :, :, :]
            examp = torch.from_numpy(examp)
            examp = examp.type(torch.FloatTensor)

            input = Variable(examp)
            label_in = train_labels[ins:(batches + ins)]
            label_in = [int(x) for x in label_in]
            label_in = torch.LongTensor(label_in)
            label_in = Variable(label_in)

            optimizer.zero_grad()

            output = cnn(input)
            loss = criterion(output, label_in)
            loss.backward()
            optimizer.step()

            print(loss.data[0])
