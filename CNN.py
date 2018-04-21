"""
Starting Net.

Working, ready for data input, need to look into batch minipulation (multiple dim output)
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    """
    Basic CNN.

    Using as basis for evantual netork
    Input of form:
    Batch Size - Channels - Height - Width
    """

    def __init__(self, batch_size,
                 channels, height,
                 width, output_size):
        """
        Instances.

        Might make function in order to most accurately create Dense layers
        """
        super(Net, self).__init__()
        conv1_shape = 5
        pool_1_shape = 2
        pool_2_shape = 2
        conv1_outshape = 10
        conv2_outshape = 20
        self.output = output_size
        self.conv1 = nn.Conv2d(channels,
                               conv1_outshape,
                               conv1_shape)
        self.maxpool1 = nn.MaxPool2d(pool_1_shape)
        self.conv2 = nn.Conv2d(conv1_outshape,
                               conv2_outshape,
                               conv1_shape)
        self.maxpool2 = nn.MaxPool2d(pool_2_shape)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        """
        Forward.

        Moving the image through the convulutions
        have two maxpools for changing shape
        """
        inner = F.relu(self.conv1(input))
        inner = self.maxpool1(inner)
        inner = F.relu(self.conv2(inner))
        inner = self.maxpool2(inner)
        inner = self.flatten(inner)
        dense_1 = nn.Linear(inner.size()[1], int(inner.size()[1] / 100))
        dense_2 = nn.Linear(int(inner.size()[1] / 100), self.output)
        inner = F.relu(dense_1(inner))
        inner = F.relu(dense_2(inner))
        output = inner
        return output

    def flatten(self, input):
        """
        Flatten.

        Flatten out the batches for densely connected layers
        """
        in_shape = input.size()[1] * input.size()[2] * input.size()[3]
        output = input.view(-1, in_shape)
        return output

    def fully_connected(self, input):
        """
        State how fully connected moves.

        Sure.
        """
        layers = 10
        output = input
        for x in range(layers):
            in_shape = input.size()[0]
            out_shape = int(in_shape / 5)
            if out_shape < self.output:
                out_shape = self.output
            dense = nn.Linear(in_shape, out_shape)
            output = F.sigmoid(dense(input))
            if out_shape == self.output:
                return output
            in_shape = out_shape


def main():
    """Placeholder."""
    batches = 30
    channels = 2
    h = 150
    w = 150
    label_size = 10

    cnn = Net(batches, channels, h, w, label_size)

    input = torch.randn(batches, channels, h, w)
    input = Variable(input)
    print(input)
    criterion = nn.CrossEntropyLoss()
    label = torch.LongTensor(np.random.randint(10, size=batches))
    label = Variable(label)

    optimizer = optim.SGD(cnn.parameters(),
                          lr=0.001,
                          momentum=0.9)

    optimizer.zero_grad()

    output = cnn(input)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()

    print(loss.data[0])

if __name__ == '__main__':
    main()
