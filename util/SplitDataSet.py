# -*- coding= utf-8 -*-
# @Author : 尘小风
# @File : SplitDtaSet.py
# @software : PyCharm

"""
用于数据集的拆分，按照6：2：2比例划分成：训练集、验证集和测试集
"""
import os
from shutil import copy
import random

# 创建保存目录
def make_file(file):
    if not os.path.exists(file):
        os.makedirs(file)

# 获取 cell 文件夹下除 .txt文件以外所有文件夹名（即5种宫颈细胞的类名）
file_path = "../dataset/SIPaKMeD/cell/"
cell_class = [cla for cla in os.listdir(file_path) if ".txt" not in cla]

# 创建训练集train文件夹，并由5种类名在其目录下创建5个子目录
make_file('../dataset/SIPaKMeD/train')
for cla in cell_class:
    make_file('../dataset/SIPaKMeD/train/' + cla)

# 创建验证集val文件夹，并由5种类名在其目录下创建5个子目录
make_file('../dataset/SIPaKMeD/val')
for cla in cell_class:
    make_file('../dataset/SIPaKMeD/val/' + cla)

# 创建测试集test文件夹，并由5种类名在其目录下创建5个子目录
make_file('../dataset/SIPaKMeD/test')
for cla in cell_class:
    make_file('../dataset/SIPaKMeD/test/' + cla)

# 划分比例6：2：2
# 先从总数据中划出0.4作为训练和验证，在从这0.4里划出一半作为测试，剩下一半作为验证
split_rate = 0.4
split_rate_test = 0.5

# 遍历5种细胞的全部图像并按比例分成训练集和测试集
for cla in cell_class:
    cla_path = file_path + '/' + cla + '/'  # 某一类别细胞的子目录
    img_name = os.listdir(cla_path)  # img_name列表存储了该目录下所有图像的名称
    num = len(img_name)
    testAndVal_index = random.sample(img_name, k=int(num * split_rate))
    num_testAndVal = len(testAndVal_index)
    test_index = random.sample(testAndVal_index, k=int(num_testAndVal * split_rate_test))

    for index, image in enumerate(img_name):
        # testAndVal_index中保存测试集 + 验证集的图像名称
        if image in testAndVal_index:
            for sub_index, sub_image in enumerate(testAndVal_index):
                if sub_image in test_index:
                    image_path = cla_path + sub_image
                    new_path = '../dataset/SIPaKMeD/test/' + cla
                    copy(image_path, new_path)  # 将选中的图像复制到新路径
                else:
                    image_path = cla_path + sub_image
                    new_path = '../dataset/SIPaKMeD/val/' + cla
                    copy(image_path, new_path)

        # 其余的图像保存在训练集中
        else:
            image_path = cla_path + image
            new_path = '../dataset/SIPaKMeD/train/' + cla
            copy(image_path, new_path)

        # 实时显示拆分进度
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")
    print()


print("processing done!")
