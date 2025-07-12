# -*- coding= utf-8 -*-
# @Time : 2023/3/30 10:03
# @Author : 尘小风
# @File : Model.py
# @software : PyCharm

import torch

# 把卷积层和激活函数打包成一个基础的卷积模块
class BasicConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, **kwargs)
        self.active = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.active(x)
        return x


# 定义Inception模块
class Inception(torch.nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3reduce, ch3x3, ch5x5reduce, ch5x5, ch_pool):
        super(Inception, self).__init__()
        # 第一个分支
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        # 第二个分支
        self.branch2 = torch.nn.Sequential(
            BasicConv2d(in_channels, ch3x3reduce, kernel_size=1),
            BasicConv2d(ch3x3reduce, ch3x3, kernel_size=3, padding=1, stride=1),
        )
        # 第三个分支
        self.branch3 = torch.nn.Sequential(
            BasicConv2d(in_channels, ch5x5reduce, kernel_size=1),
            BasicConv2d(ch5x5reduce, ch5x5, kernel_size=5, padding=2, stride=1),
        )
        # 第四个分支
        self.branch4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            BasicConv2d(in_channels, ch_pool, kernel_size=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        # [batch, channels, H, W] 从channels维度进行拼接
        return torch.cat(outputs, dim=1)


# 定义辅助分类器
class InceptionAux(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagepool = torch.nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        self.fc1 = torch.nn.Linear(128*4*4, 1024)
        self.fc2 = torch.nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.averagepool(x)
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        # 在model.train()模式下self.training=True，在model.eval()模式下self.training=False，为Ture时执行dropout，为False时不执行
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x


class Model(torch.nn.Module):
    def __init__(self, num_classes, aux_logits=True, init_weights=False): # aux_logits=True表示使用辅助分类器
        super(Model, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True) # ceil_mode=True代表计算结果向上取整

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.incep3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.incep3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.incep4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.incep4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.incep4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.incep4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.incep4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.incep5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.incep5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logits: # 如果传入的aux_logits为True，则定义self.aux1和self.aux2
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.averagepool = torch.nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = torch.nn.Dropout(0.4)
        self.fc = torch.nn.Linear(1024, num_classes)

        if init_weights: # 如果传入的init_weight为True，则采用_initialize_weights()的参数初始化方法
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.incep3a(x)
        x = self.incep3b(x)
        x = self.maxpool3(x)
        x = self.incep4a(x)

        if self.training and self.aux_logits: # 如果是训练过程，并且使用辅助分类器
            aux1 = self.aux1(x)

        x = self.incep4b(x)
        x = self.incep4c(x)
        x = self.incep4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        x = self.incep4e(x)
        x = self.maxpool4(x)
        x = self.incep5a(x)
        x = self.incep5b(x)

        x = self.averagepool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.aux_logits:
            return x, aux2, aux1

        return x # 如果不是训练过程，或者不使用辅助分类器，则直接返回x

    def _initialize_weights(self):
        for m in self.modules():  # 遍历每一层网络结构
            if isinstance(m, torch.nn.Conv2d):  # 如果是卷积层
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # 何恺明初始化方法
                # torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:  # 如果有偏置参数
                    torch.nn.init.constant_(m.bias, 0)  # 把偏置参数初始化为0
            elif isinstance(m, torch.nn.Linear):  # 如果是全连接层
                # torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)  # 把偏置参数初始化为0


def googlenet(num_classes=1000, aux_logits=True, init_weights=True):
    return Model(num_classes=num_classes, aux_logits=True, init_weights=True)





