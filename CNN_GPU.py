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
        #conv1_shape = (height, 2)
        conv1_shape = 5
        conv2_shape = 3
        pool_1_shape = 3
        pool_2_shape = 3
        conv1_outshape = 50
        conv2_outshape = 60
        self.output = output_size
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=channels,
                                             out_channels=conv1_outshape,
                                             kernel_size=conv1_shape,
                                             stride=3,
                                             padding=1),
                                   nn.BatchNorm2d(conv1_outshape),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=pool_1_shape))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=conv1_outshape,
                                             out_channels=conv2_outshape,
                                             kernel_size=conv2_shape,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(conv2_outshape),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=pool_2_shape))
        # self.dropout1 = nn.Dropout(p=0.2)
        self.lin4 = nn.Sequential(nn.Linear(60*4*23, 4096),
                                  nn.Linear(4096, 4096),
                                  nn.Linear(4096, 4096),
                                  nn.Linear(4096, 256),
                                  nn.Linear(256, output_size))

    def forward(self, input):
        """
        Forward.

        Moving the image through the convulutions
        have two maxpools for changing shape
        """
        inner = self.conv1(input)
        inner = self.conv2(inner)
        # print(inner.size())
        # exit()
        inner = inner.view(inner.size(0), -1)
        output = self.lin4(inner)
        return output
