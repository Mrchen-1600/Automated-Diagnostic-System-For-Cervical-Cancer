# -*- coding= utf-8 -*-
# @Time : 2023/3/19 9:54
# @Author : 尘小风
# @File : Model.py
# @software : PyCharm

import torch

class Model(torch.nn.Module):
    def __init__(self, num_classes): # 传入所需分类的类别数，也就是最后一个全连接层所需要输出的channel
        super(Model, self).__init__()
        self.features = torch.nn.Sequential( # 定义卷积层提取图像特征
            # 计算output的size的计算公式：(input_size-kernel_size+padding)/stride + 1
            torch.nn.Conv2d(3, 48, kernel_size=11, padding=2, stride=4),
            torch.nn.ReLU(inplace=True), # 直接修改覆盖原值，节省运算内存
            torch.nn.MaxPool2d(kernel_size=3, padding=0, stride=2),
            torch.nn.Conv2d(48, 128, kernel_size=5, padding=2, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, padding=0, stride=2),
            torch.nn.Conv2d(128, 192, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(192, 128, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, padding=0, stride=2),
        )

        self.classifier = torch.nn.Sequential( # 定义全连接层进行图像分类
            # dropout随机失活神经元，默认比例0.5，防止过拟合 提升模型泛化能力。
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128*6*6, 2048) ,
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2048, num_classes),
        )

    def forward(self, x):
       x = self.features(x)
       x = torch.flatten(x, start_dim=1) # (batch_size, c , H, W)即从通道c开始对x进行展平
       x = self.classifier(x)
       return x

def alexnet(num_classes=1000):
    return Model(num_classes=num_classes)