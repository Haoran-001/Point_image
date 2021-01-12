import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .Model import Model
from torchsummary import summary


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class SVCNN(Model):
    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='vgg11'):
        super(SVCNN, self).__init__(name)

        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.use_densenet = cnn_name.startswith('desnet')

        self.mean = Variable(torch.FloatTensor([0.0142, 0.0142, 0.0142]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.0818, 0.0818, 0.0818]), requires_grad=False).cuda()

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, 40)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, 40)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, 40)
            elif self.cnn_name == 'resnet50x':
                self.net = models.resnext50_32x4d(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, 40)
            elif self.cnn_name == 'resnet101x':
                self.net = models.resnext101_32x8d(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, 40)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretraining).features
                self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier

            self.net_2._modules['6'] = nn.Linear(4096, 40)

    def forward(self, x):
        # print(x.size())
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            # print(y.size())
            return self.net_2(y.view(y.shape[0], -1))


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

        self.mean = Variable(torch.FloatTensor([0.0142, 0.0142, 0.0142]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.0818, 0.0818, 0.0818]), requires_grad=False).cuda()

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = nn.Sequential(nn.Linear(4096, 40))
            self.conv = nn.Conv2d(1, self.num_conv, (1, 12), padding_mode='valid')
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2

        self.mp3 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.ap3 = nn.AvgPool3d((1, 7, 7), stride=(1, 1, 1))
        padding = (1, 1, 1)
        self.conv1 = nn.Sequential(nn.BatchNorm3d(64), nn.ReLU(inplace=True),
                                   nn.Conv3d(64, 32, (3, 3, 3), bias=False, padding=padding))
        self.conv2 = nn.Sequential(nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                                   nn.Conv3d(32, 32, (3, 3, 3), padding=padding, bias=False))
        self.conv3 = nn.Sequential(nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                                   nn.Conv3d(32, 64, (3, 3, 3), padding=padding, bias=False))
        self.conv4 = nn.Sequential(nn.BatchNorm3d(64), nn.ReLU(inplace=True),
                                   nn.Conv3d(64, 64, (3, 3, 3), padding=padding, bias=False))
        self.conv5 = nn.Sequential(nn.BatchNorm3d(64), nn.ReLU(inplace=True),
                                   nn.Conv3d(64, 128, (3, 3, 3), padding=padding, bias=False))
        self.conv6 = nn.Sequential(nn.BatchNorm3d(128), nn.ReLU(inplace=True),
                                   nn.Conv3d(128, 128, (3, 3, 3), padding=padding, bias=False))
        self.conv7 = nn.Sequential(nn.BatchNorm3d(128), nn.ReLU(inplace=True),
                                   nn.Conv3d(128, 256, (3, 3, 3), padding=padding, bias=False))
        self.conv8 = nn.Sequential(nn.BatchNorm3d(256), nn.ReLU(inplace=True),
                                   nn.Conv3d(256, 256, (3, 3, 3), padding=padding, bias=False))
        self.shortcut1 = nn.Sequential(nn.BatchNorm3d(64), nn.ReLU(inplace=True),
                                       nn.Conv3d(64, 32, (1, 1, 1), bias=False))
        self.shortcut2 = nn.Sequential(nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 64, (1, 1, 1), bias=False))
        self.shortcut3 = nn.Sequential(nn.BatchNorm3d(64), nn.ReLU(inplace=True),
                                       nn.Conv3d(64, 128, (1, 1, 1), bias=False))
        self.shortcut4 = nn.Sequential(nn.BatchNorm3d(128), nn.ReLU(inplace=True),
                                       nn.Conv3d(128, 256, (1, 1, 1), bias=False))

    def forward(self, x):
        # x_0 = self.net_2(x)
        # x_0 = x_0.view(int(x_0.shape[0] / self.num_views), x_0.shape[1], self.num_views, x_0.shape[-2], x_0.shape[-1])
        # x_1 = self.conv1(x_0)
        # x_1 = self.conv2(x_1)
        # x_1 += self.shortcut1(x_0)
        # x_1 = self.mp3(x_1)
        # # print(x_1.size())
        # x_2 = self.conv3(x_1)
        # x_2 = self.conv4(x_2)
        # x_2 += self.shortcut2(x_1)
        # x_2 = self.mp3(x_2)
        # # print(x_2.size())
        # x_3 = self.conv5(x_2)
        # x_3 = self.conv6(x_3)
        # x_3 += self.shortcut3(x_2)
        # x_3 = self.mp3(x_3)
        # # print(x_3.size())
        # x_4 = self.conv7(x_3)
        # x_4 = self.conv8(x_4)
        # x_4 += self.shortcut4(x_3)
        # x_4 = self.ap3(x_4)
        # # print(x_4.size())
        # y_view = x_4.view(x_4.shape[0], x_4.shape[1]*x_4.shape[2])
        # # print(y_view.size())
        y = self.net_1(x)
        print('y:', y.size())

        ### self attention
        y = y.view((int(x.shape[0] / self.num_views), 1, y.shape[-3], self.num_views))
        # print('the reshape of y:', y.size())
        conv_y = self.conv(y).view(y.shape[0], y.shape[2], self.num_conv)
        # print('conv_y', conv_y.size())
        y1 = torch.max(y, 3)[0]
        y2 = torch.mean(y, 3)
        cat_y = torch.cat((y1, y2), 0).view(y.shape[0], 2, -1)
        # print('cat_y', cat_y.size())
        cat_y = self.class_conv(cat_y).view(y.shape[0], -1)
        # print('cat_y', cat_y.size())
        scale = nn.functional.softmax(cat_y, dim=1).view(y.shape[0], 2, 1)
        # print('scale', scale.size())
        y = torch.matmul(conv_y, scale)
        # print('matmul', y.size())
        y = y.view(y.shape[0], -1)
        return self.net_2(y)


if __name__ == '__main__':
    cnn_name = 'resnet50'
    name = 'MVCNN'
    cnet = SVCNN(name, nclasses=40, pretraining=True, cnn_name=cnn_name)
    model = MVCNN(name, cnet, nclasses=40, cnn_name=cnn_name, num_views=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    summary(model, input_size=(3, 224, 224))
