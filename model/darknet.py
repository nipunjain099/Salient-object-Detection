import torch
import torch.nn as nn
import numpy as np

class Darknet19(nn.Module):
    # VGG16 with two branches
    # pooling layer at the front of block
    def __init__(self):
        super(Darknet19, self).__init__()
        conv1 = nn.Sequential()
        conv1.add_module('conv1_1', nn.Conv2d(3, 32, 3, 1, 1, bias=False))
        conv1.add_module('bn1_1', nn.BatchNorm2d(32))
        conv1.add_module('relu1_1', nn.LeakyReLU(0.1, inplace=True))
        self.conv1 = conv1

        conv2 = nn.Sequential()
        conv2.add_module('maxpool1', nn.MaxPool2d(2, stride=2))
        conv2.add_module('conv2_1', nn.Conv2d(32, 64, 3, 1, 1, bias=False))
        conv2.add_module('bn2_1', nn.BatchNorm2d(64))
        conv2.add_module('relu2_1', nn.LeakyReLU(0.1, inplace=True))
        self.conv2 = conv2

        conv3 = nn.Sequential()
        conv3.add_module('maxpool2', nn.MaxPool2d(2, stride=2))
        conv3.add_module('conv3_1', nn.Conv2d(64, 128, 3, 1, 1, bias=False))
        conv3.add_module('bn3_1', nn.BatchNorm2d(128))
        conv3.add_module('relu3_1', nn.LeakyReLU(0.1, inplace=True))
        conv3.add_module('conv3_2', nn.Conv2d(128, 64, 1, 1, 0, bias=False))
        conv3.add_module('bn3_2', nn.BatchNorm2d(64))
        conv3.add_module('relu3_2', nn.LeakyReLU(0.1, inplace=True))
        conv3.add_module('conv3_3', nn.Conv2d(64, 128, 3, 1, 1, bias=False))
        conv3.add_module('bn3_3', nn.BatchNorm2d(128))
        conv3.add_module('relu3_3', nn.LeakyReLU(0.1, inplace=True))
        self.conv3 = conv3

        conv4_1 = nn.Sequential()
        conv4_1.add_module('maxpool4_1', nn.MaxPool2d(2, stride=2))
        conv4_1.add_module('conv4_1_1', nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        conv4_1.add_module('bn4_1_1', nn.BatchNorm2d(256))
        conv4_1.add_module('relu4_1_1', nn.LeakyReLU(0.1, inplace=True))
        conv4_1.add_module('conv4_1_2', nn.Conv2d(256, 128, 1, 1, 0, bias=False))
        conv4_1.add_module('bn4_1_2', nn.BatchNorm2d(128))
        conv4_1.add_module('relu4_1_2', nn.LeakyReLU(0.1, inplace=True))
        conv4_1.add_module('conv4_1_3', nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        conv4_1.add_module('bn4_1_3', nn.BatchNorm2d(256))
        conv4_1.add_module('relu4_1_3', nn.LeakyReLU(0.1, inplace=True))
        self.conv4_1 = conv4_1

        conv5_1 = nn.Sequential()
        conv5_1.add_module('maxpool5_1', nn.MaxPool2d(2, stride=2))
        conv5_1.add_module('conv5_1_1', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        conv5_1.add_module('bn5_1_1', nn.BatchNorm2d(512))
        conv5_1.add_module('relu5_1_1', nn.LeakyReLU(0.1, inplace=True))
        conv5_1.add_module('conv5_1_2', nn.Conv2d(512, 256, 1, 1, 0, bias=False))
        conv5_1.add_module('bn5_1_2', nn.BatchNorm2d(256))
        conv5_1.add_module('relu5_1_2', nn.LeakyReLU(0.1, inplace=True))
        conv5_1.add_module('conv5_1_3', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        conv5_1.add_module('bn5_1_3', nn.BatchNorm2d(512))
        conv5_1.add_module('relu5_1_3', nn.LeakyReLU(0.1, inplace=True))
        conv5_1.add_module('conv5_1_4', nn.Conv2d(512, 256, 1, 1, 0, bias=False))
        conv5_1.add_module('bn5_1_4', nn.BatchNorm2d(256))
        conv5_1.add_module('relu5_1_4', nn.LeakyReLU(0.1, inplace=True))
        conv5_1.add_module('conv5_1_5', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        conv5_1.add_module('bn5_1_5', nn.BatchNorm2d(512))
        conv5_1.add_module('relu5_1_5', nn.LeakyReLU(0.1, inplace=True))
        self.conv5_1 = conv5_1

        conv4_2 = nn.Sequential()
        conv4_2.add_module('maxpool4_2', nn.MaxPool2d(2, stride=2))
        conv4_2.add_module('conv4_2_1', nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        conv4_2.add_module('bn4_2_1', nn.BatchNorm2d(256))
        conv4_2.add_module('relu4_2_1', nn.LeakyReLU(0.1, inplace=True))
        conv4_2.add_module('conv4_2_2', nn.Conv2d(256, 128, 1, 1, 0, bias=False))
        conv4_2.add_module('bn4_2_2', nn.BatchNorm2d(128))
        conv4_2.add_module('relu4_2_2', nn.LeakyReLU(0.1, inplace=True))
        conv4_2.add_module('conv4_2_3', nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        conv4_2.add_module('bn4_2_3', nn.BatchNorm2d(256))
        conv4_2.add_module('relu4_2_3', nn.LeakyReLU(0.1, inplace=True))
        self.conv4_2 = conv4_2

        conv5_2 = nn.Sequential()
        conv5_2.add_module('maxpool5_1', nn.MaxPool2d(2, stride=2))
        conv5_2.add_module('conv5_2_1', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        conv5_2.add_module('bn5_2_1', nn.BatchNorm2d(512))
        conv5_2.add_module('relu5_2_1', nn.LeakyReLU(0.1, inplace=True))
        conv5_2.add_module('conv5_2_2', nn.Conv2d(512, 256, 1, 1, 0, bias=False))
        conv5_2.add_module('bn5_2_2', nn.BatchNorm2d(256))
        conv5_2.add_module('relu5_2_2', nn.LeakyReLU(0.1, inplace=True))
        conv5_2.add_module('conv5_2_3', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        conv5_2.add_module('bn5_2_3', nn.BatchNorm2d(512))
        conv5_2.add_module('relu5_2_3', nn.LeakyReLU(0.1, inplace=True))
        conv5_2.add_module('conv5_2_4', nn.Conv2d(512, 256, 1, 1, 0, bias=False))
        conv5_2.add_module('bn5_2_4', nn.BatchNorm2d(256))
        conv5_2.add_module('relu5_2_4', nn.LeakyReLU(0.1, inplace=True))
        conv5_2.add_module('conv5_2_5', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        conv5_2.add_module('bn5_2_5', nn.BatchNorm2d(512))
        conv5_2.add_module('relu5_2_5', nn.LeakyReLU(0.1, inplace=True))
        self.conv5_2 = conv5_2

        weights = torch.load('darknet19.pth')
        self._initialize_weights(weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x1 = self.conv4_1(x)
        x1 = self.conv5_1(x1)
        x2 = self.conv4_2(x)
        x2 = self.conv5_2(x2)
        return x1, x2

    def _initialize_weights(self, weights):
        keys = list(weights.keys())
        i = 0
        self.conv1.conv1_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv1.bn1_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv1.bn1_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv1.bn1_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv1.bn1_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv1.bn1_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv2.conv2_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv3.conv3_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.conv3_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.conv3_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1


        self.conv4_1.conv4_1_1.weight.data.copy_(weights[keys[i]])
        self.conv4_2.conv4_2_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_1.weight.data.copy_(weights[keys[i]])
        self.conv4_2.bn4_2_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_1.bias.data.copy_(weights[keys[i]])
        self.conv4_2.bn4_2_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_1.running_mean.data.copy_(weights[keys[i]])
        self.conv4_2.bn4_2_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_1.running_var.data.copy_(weights[keys[i]])
        self.conv4_2.bn4_2_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_1.num_batches_tracked.data.copy_(weights[keys[i]])
        self.conv4_2.bn4_2_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.conv4_1_2.weight.data.copy_(weights[keys[i]])
        self.conv4_2.conv4_2_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_2.weight.data.copy_(weights[keys[i]])
        self.conv4_2.bn4_2_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_2.bias.data.copy_(weights[keys[i]])
        self.conv4_2.bn4_2_2.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_2.running_mean.data.copy_(weights[keys[i]])
        self.conv4_2.bn4_2_2.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_2.running_var.data.copy_(weights[keys[i]])
        self.conv4_2.bn4_2_2.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_2.num_batches_tracked.data.copy_(weights[keys[i]])
        self.conv4_2.bn4_2_2.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.conv4_1_3.weight.data.copy_(weights[keys[i]])
        self.conv4_2.conv4_2_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_3.weight.data.copy_(weights[keys[i]])
        self.conv4_2.bn4_2_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_3.bias.data.copy_(weights[keys[i]])
        self.conv4_2.bn4_2_3.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_3.running_mean.data.copy_(weights[keys[i]])
        self.conv4_2.bn4_2_3.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_3.running_var.data.copy_(weights[keys[i]])
        self.conv4_2.bn4_2_3.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_3.num_batches_tracked.data.copy_(weights[keys[i]])
        self.conv4_2.bn4_2_3.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv5_1.conv5_1_1.weight.data.copy_(weights[keys[i]])
        self.conv5_2.conv5_2_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_1.weight.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_1.bias.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_1.running_mean.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_1.running_var.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_1.num_batches_tracked.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.conv5_1_2.weight.data.copy_(weights[keys[i]])
        self.conv5_2.conv5_2_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_2.weight.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_2.bias.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_2.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_2.running_mean.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_2.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_2.running_var.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_2.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_2.num_batches_tracked.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_2.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.conv5_1_3.weight.data.copy_(weights[keys[i]])
        self.conv5_2.conv5_2_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_3.weight.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_3.bias.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_3.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_3.running_mean.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_3.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_3.running_var.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_3.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_3.num_batches_tracked.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_3.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.conv5_1_4.weight.data.copy_(weights[keys[i]])
        self.conv5_2.conv5_2_4.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_4.weight.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_4.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_4.bias.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_4.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_4.running_mean.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_4.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_4.running_var.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_4.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_4.num_batches_tracked.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_4.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.conv5_1_5.weight.data.copy_(weights[keys[i]])
        self.conv5_2.conv5_2_5.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_5.weight.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_5.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_5.bias.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_5.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_5.running_mean.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_5.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_5.running_var.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_5.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_5.num_batches_tracked.data.copy_(weights[keys[i]])
        self.conv5_2.bn5_2_5.num_batches_tracked.data.copy_(weights[keys[i]])
        
class Darknet19_A(nn.Module):
    # VGG16 with two branches
    # pooling layer at the front of block
    def __init__(self):
        super(Darknet19_A, self).__init__()
        conv1 = nn.Sequential()
        conv1.add_module('conv1_1', nn.Conv2d(3, 32, 3, 1, 1, bias=False))
        conv1.add_module('bn1_1', nn.BatchNorm2d(32))
        conv1.add_module('relu1_1', nn.LeakyReLU(0.1, inplace=True))
        self.conv1 = conv1

        conv2 = nn.Sequential()
        conv2.add_module('maxpool1', nn.MaxPool2d(2, stride=2))
        conv2.add_module('conv2_1', nn.Conv2d(32, 64, 3, 1, 1, bias=False))
        conv2.add_module('bn2_1', nn.BatchNorm2d(64))
        conv2.add_module('relu2_1', nn.LeakyReLU(0.1, inplace=True))
        self.conv2 = conv2

        conv3 = nn.Sequential()
        conv3.add_module('maxpool2', nn.MaxPool2d(2, stride=2))
        conv3.add_module('conv3_1', nn.Conv2d(64, 128, 3, 1, 1, bias=False))
        conv3.add_module('bn3_1', nn.BatchNorm2d(128))
        conv3.add_module('relu3_1', nn.LeakyReLU(0.1, inplace=True))
        conv3.add_module('conv3_2', nn.Conv2d(128, 64, 1, 1, 0, bias=False))
        conv3.add_module('bn3_2', nn.BatchNorm2d(64))
        conv3.add_module('relu3_2', nn.LeakyReLU(0.1, inplace=True))
        conv3.add_module('conv3_3', nn.Conv2d(64, 128, 3, 1, 1, bias=False))
        conv3.add_module('bn3_3', nn.BatchNorm2d(128))
        conv3.add_module('relu3_3', nn.LeakyReLU(0.1, inplace=True))
        self.conv3 = conv3

        conv4_1 = nn.Sequential()
        conv4_1.add_module('maxpool4_1', nn.MaxPool2d(2, stride=2))
        conv4_1.add_module('conv4_1_1', nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        conv4_1.add_module('bn4_1_1', nn.BatchNorm2d(256))
        conv4_1.add_module('relu4_1_1', nn.LeakyReLU(0.1, inplace=True))
        conv4_1.add_module('conv4_1_2', nn.Conv2d(256, 128, 1, 1, 0, bias=False))
        conv4_1.add_module('bn4_1_2', nn.BatchNorm2d(128))
        conv4_1.add_module('relu4_1_2', nn.LeakyReLU(0.1, inplace=True))
        conv4_1.add_module('conv4_1_3', nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        conv4_1.add_module('bn4_1_3', nn.BatchNorm2d(256))
        conv4_1.add_module('relu4_1_3', nn.LeakyReLU(0.1, inplace=True))
        self.conv4_1 = conv4_1

        conv5_1 = nn.Sequential()
        conv5_1.add_module('maxpool5_1', nn.MaxPool2d(2, stride=2))
        conv5_1.add_module('conv5_1_1', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        conv5_1.add_module('bn5_1_1', nn.BatchNorm2d(512))
        conv5_1.add_module('relu5_1_1', nn.LeakyReLU(0.1, inplace=True))
        conv5_1.add_module('conv5_1_2', nn.Conv2d(512, 256, 1, 1, 0, bias=False))
        conv5_1.add_module('bn5_1_2', nn.BatchNorm2d(256))
        conv5_1.add_module('relu5_1_2', nn.LeakyReLU(0.1, inplace=True))
        conv5_1.add_module('conv5_1_3', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        conv5_1.add_module('bn5_1_3', nn.BatchNorm2d(512))
        conv5_1.add_module('relu5_1_3', nn.LeakyReLU(0.1, inplace=True))
        self.conv5_1 = conv5_1

        weights = torch.load('darknet19.pth')
        self._initialize_weights(weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x1 = self.conv4_1(x)
        x1 = self.conv5_1(x1)
        return x1

    def _initialize_weights(self, weights):
        keys = list(weights.keys())
        i = 0
        self.conv1.conv1_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv1.bn1_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv1.bn1_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv1.bn1_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv1.bn1_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv1.bn1_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv2.conv2_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv2.bn2_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv3.conv3_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.conv3_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_2.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.conv3_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv3.bn3_3.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv4_1.conv4_1_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.conv4_1_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_2.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_2.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_2.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_2.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.conv4_1_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_3.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_3.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_3.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv4_1.bn4_1_3.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv5_1.conv5_1_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_1.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_1.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_1.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_1.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_1.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.conv5_1_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_2.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_2.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_2.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_2.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_2.num_batches_tracked.data.copy_(weights[keys[i]])
        i+=1

        self.conv5_1.conv5_1_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_3.weight.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_3.bias.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_3.running_mean.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_3.running_var.data.copy_(weights[keys[i]])
        i+=1
        self.conv5_1.bn5_1_3.num_batches_tracked.data.copy_(weights[keys[i]])

class Darknet_A(nn.Module):
    # VGG16 with two branches
    # pooling layer at the front of block
    def __init__(self):
        super(Darknet_A, self).__init__()
        conv1 = nn.Sequential()
        conv1.add_module('conv1_1', nn.Conv2d(3, 16, 3, 1, 1, bias=False))
        conv1.add_module('bn1_1', nn.BatchNorm2d(16))
        conv1.add_module('relu1_1', nn.LeakyReLU(0.1, inplace=True))
        self.conv1 = conv1

        conv2 = nn.Sequential()
        conv2.add_module('maxpool1', nn.MaxPool2d(2, stride=2))
        conv2.add_module('conv2_1', nn.Conv2d(16, 32, 3, 1, 1, bias=False))
        conv2.add_module('bn2_1', nn.BatchNorm2d(32))
        conv2.add_module('relu2_1', nn.LeakyReLU(0.1, inplace=True))
        self.conv2 = conv2

        conv3 = nn.Sequential()
        conv3.add_module('maxpool2', nn.MaxPool2d(2, stride=2))
        conv3.add_module('conv3_1', nn.Conv2d(32, 64, 3, 1, 1, bias=False))
        conv3.add_module('bn3_1', nn.BatchNorm2d(64))
        conv3.add_module('relu3_1', nn.LeakyReLU(0.1, inplace=True))
        conv3.add_module('maxpool2', nn.MaxPool2d(2, stride=2))
        conv3.add_module('conv3_2', nn.Conv2d(64, 128, 3, 1, 1, bias=False))
        conv3.add_module('bn3_2', nn.BatchNorm2d(128))
        conv3.add_module('relu3_2', nn.LeakyReLU(0.1, inplace=True))
        self.conv3 = conv3

        conv4_1 = nn.Sequential()
        conv4_1.add_module('maxpool4_1', nn.MaxPool2d(2, stride=2))
        conv4_1.add_module('conv4_1_1', nn.Conv2d(128, 256, 3, 1, 1, bias=False))
        conv4_1.add_module('bn4_1_1', nn.BatchNorm2d(256))
        conv4_1.add_module('relu4_1_1', nn.LeakyReLU(0.1, inplace=True))
        self.conv4_1 = conv4_1

        conv5_1 = nn.Sequential()
        conv5_1.add_module('maxpool5_1', nn.MaxPool2d(2, stride=2))
        conv5_1.add_module('conv5_1_1', nn.Conv2d(256, 512, 3, 1, 1, bias=False))
        conv5_1.add_module('bn5_1_1', nn.BatchNorm2d(512))
        conv5_1.add_module('relu5_1_1', nn.LeakyReLU(0.1, inplace=True))
        self.conv5_1 = conv5_1

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x1 = self.conv4_1(x)
        x1 = self.conv5_1(x1)
        return x1

if __name__ == '__main__':

    model = Darknet19_m()
