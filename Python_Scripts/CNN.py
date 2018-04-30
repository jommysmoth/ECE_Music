"""
Starting Net.

Working, ready for data input, need to look into batch minipulation (multiple dim output)
"""
import torch.nn as nn
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
        conv1_shape = (height, 1)
        conv2_shape = 5
        pool_1_shape = 2
        pool_2_shape = 2
        conv1_outshape = 20
        conv2_outshape = 40
        self.output = output_size
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=channels,
                                             out_channels=conv1_outshape,
                                             kernel_size=conv1_shape,
                                             stride=1,
                                             padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=pool_1_shape))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=conv1_outshape,
                                             out_channels=conv2_outshape,
                                             kernel_size=conv2_shape,
                                             stride=1,
                                             padding=2),
                                   nn.BatchNorm2d(conv2_outshape),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=pool_2_shape))
        self.dropout1 = nn.Dropout(p=0.2)
        self.dense_1 = nn.Linear(40 * 81, 1024)
        self.dense_mid = nn.Linear(1024, 256)
        self.dense_out = nn.Linear(256, output_size)
        self.lin4 = nn.Sequential(nn.Linear(40 * 81, 1024),
                                  nn.Linear(1024, 256),
                                  nn.Linear(256, output_size))

    def forward(self, input):
        """
        Forward.

        Moving the image through the convulutions
        have two maxpools for changing shape
        """
        inner = self.conv1(input)
        print(inner.size())
        inner = self.conv2(inner)
        print(inner.size())
        # exit()
        inner = inner.view(inner.size(0), -1)
        """
        inner = self.dropout1(self.dense_1(inner))
        inner = self.dense_mid(inner)
        output = self.dense_out(inner)
        """
        output = self.lin4(inner)
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
