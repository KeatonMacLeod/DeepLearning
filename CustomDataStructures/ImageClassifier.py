import torch
import torchvision
import os
from os import listdir
from os.path import isfile, join
from torch.utils import data
import torch.nn as nn
import torch.functional as F
from torchvision import transforms
from CustomDataStructures.ImageDataset import ImageDataset


class ImageClassifier(nn.Module):

    def __init__(self):
        super(ImageClassifier, self).__init__()

        # Transforms the images into a representation which can be used for training and testing -> investigate why
        # we are using these sizes for these datasets later
        self.transform_image = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])])

        # A convolution layer convolves the input tensors into a more abstract / higher-level representation which help
        # to combat overfitting during training which should hopefully yield better results when classifying later on
        # For a convolution layer the parameters are respectively:
        # 1.) Number of input channels -> RGB (3) | Greyscale (1)
        # 2.) Number of output channels -> analogous to the number of hidden neurons in a layer
        # 3.) Size of the nxn square convolution "sliding window"
        self.conv1 = nn.Conv2d(3, 6, 4)
        self.conv2 = nn.Conv2d(6, 15, 4)

        # A fully-connected layer is a layer that takes in all the inputs from the layer "before" it in the neural
        # network. In this case, we're going to use a Linear Layer which performs a linear transformation to the
        # inputs it's receiving from the layers before it. In this case, we're going to perform the linear transform of
        # multiplying the inputs from the network by the weight matrix
        # The operation we perform here is y = Wx + b -> where x is the input, W is the weight matrix, and b is the bias
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # We only need to define the "forward" operations in our neural network, the "backward" operations where the
    # gradients will be computed are automatically computed via autograd which is built-in to the PyTorch library
    def forward(self, x):
        # After the convolution layer creates a more "abstract" representation of the input, we can use an activation
        # function in order to add non-linearity into our network. If all we had is convolution layers without
        # interleaved activation functions, our model would always be linear, thus we interleave non-linear activation
        # functions between our convolution layer to "learn more" about our network
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)))

        # -1 means the dimensions are inferred from the other dimensions -> this is just a matrix reshape
        x = x.view(-1, self.num_flat_features(x))

        # Feed the input through our fully connected layers with non-linear activation functions in between
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # We could theoretically not have fully-connected layers in our network, because each fully-connected layer
        # has an equivalent convolution layer, but it is easier to add fully-connected layers because we can specify
        # the number of output neurons (typically the number of classes) more easily, thus we add a fully-connected
        # layer at the very end so we get a final num_classes output from our forward operation
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
