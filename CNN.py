"""
Starting Net.

Using Pytorch, unless I have some issues
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


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
        conv1_outshape = 20
        conv2_outshape = 30
        self.output = output_size
        self.conv1 = nn.Conv2d(channels,
                               conv1_outshape,
                               conv1_shape)
        self.maxpool1 = nn.MaxPool2d(pool_1_shape)
        self.conv2 = nn.Conv2d(conv1_outshape,
                               conv2_outshape,
                               conv1_shape)
        self.maxpool2 = nn.MaxPool2d(pool_2_shape)

    def forward(self, input):
        """
        Forward.

        Moving the image through the convulutions
        have two maxpools for changing shape
        """
        inner = self.conv1(input)
        inner = F.relu(inner)
        inner = self.maxpool1(inner)
        inner = self.conv2(inner)
        inner = F.relu(inner)
        inner = self.maxpool2(inner)
        in_shape = inner.view(inner.numel()).size()[0]
        mid_shape = int(in_shape / 5)
        dense_in = nn.Linear(in_shape, mid_shape)
        dense_out = nn.Linear(mid_shape, self.output)
        inner = inner.view(inner.numel())
        inner = dense_in(inner)
        inner = F.relu(inner)
        output = dense_out(inner)
        return output

if __name__ == '__main__':
    cnn = Net(1, 3, 100, 75, 10)
    input = Variable(torch.randn(1, 3, 100, 75))
    output = cnn(input)
    print(output)
