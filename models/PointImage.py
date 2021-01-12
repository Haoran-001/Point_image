import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .Model import Model
from torchsummary import summary


class MVCNN(Model):
    def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views=12):
        super(MVCNN, self).__init__(name)
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        self.nclasses = nclasses
        self.num_views = num_views
        self.use_resnet = cnn_name.startswith('resnet')
        self.num_conv = 2
        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = nn.Sequential(nn.Linear(4096, 40))
            self.conv = nn.Conv2d(1, self.num_conv, (1, 12), padding_mode='valid')

        self.first_conv = nn.Conv3d(18, 24, (1, 3, 3), padding_mode='valid', stride=(1, 2, 2))
        self.spatial_conv1 = nn.Sequential(nn.BatchNorm3d, nn.ReLU(), nn.Conv3d(24, 24, (1, 3, 3)))



    def forward(self, x):
        print(x.size())
        x = x.view(int(x.shape[0] / self.num_views), 1, self.num_views * x.shape[-3], x.shape[-2], x.shape[-1])
        print(x.size())
        x_0 = self.first_conv(x)
        x_1 = self.spatial_conv1(x_0)
        x_2 = self.spatial_conv1(torch.add(x_1, x_0))



        y = self.net_1(x)
        # print('y:', y.size())
        conv = True
        if conv:
            y = y.view((int(x.shape[0] / self.num_views), 1, y.shape[-3], self.num_views))
            y = self.conv(y).view(y.shape[0], y.shape[2] * self.num_conv)
        else:
            y = y.view((int(x.shape[0] / self.num_views), self.num_views, y.shape[-3], y.shape[-2], y.shape[-1]))
            # y:(8,12,512,7,7)
            print('the reshape of y:', y.size())
            y1 = torch.max(y, 1)[0].view(y.shape[0], -1)
            y2 = torch.mean(y, 1).view(y.shape[0], -1)
            y = torch.cat((y1, y2), 1)
            print('the cat of y', y.size())
        return self.net_2(y)


if __name__ == '__main__':
    cnn_name = 'resnet50'
    name = 'MVCNN'
    cnet = SVCNN(name, nclasses=40, pretraining=True, cnn_name=cnn_name)
    model = MVCNN(name, cnet, nclasses=40, cnn_name=cnn_name, num_views=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    summary(model, input_size=(3, 224, 224))
