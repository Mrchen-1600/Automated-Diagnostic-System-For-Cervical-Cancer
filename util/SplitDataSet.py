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

# 创建保存目录，使用 exist_ok 参数避免目录已存在时的错误
def make_file(file):
    os.makedirs(file, exist_ok=True)

# 创建数据集目录结构
def create_dataset_directories(base_dir, cell_classes):
    for dataset_type in ['train', 'val', 'test']:
        dataset_dir = os.path.join(base_dir, dataset_type)
        make_file(dataset_dir)
        for cla in cell_classes:
            class_dir = os.path.join(dataset_dir, cla)
            make_file(class_dir)

# 划分数据集
def split_dataset(file_path, base_dir, cell_classes, split_rate=0.4, split_rate_test=0.5):
    for cla in cell_classes:
        cla_path = os.path.join(file_path, cla)
        img_names = os.listdir(cla_path)
        num = len(img_names)
        # 随机打乱图像列表
        random.shuffle(img_names)
        # 计算测试集和验证集的数量
        num_test_val = int(num * split_rate)
        num_test = int(num_test_val * split_rate_test)
        test_val_images = img_names[:num_test_val]
        test_images = test_val_images[:num_test]
        val_images = test_val_images[num_test:]

        for index, image in enumerate(img_names):
            image_path = os.path.join(cla_path, image)
            if image in test_images:
                new_path = os.path.join(base_dir, 'test', cla)
            elif image in val_images:
                new_path = os.path.join(base_dir, 'val', cla)
            else:
                new_path = os.path.join(base_dir, 'train', cla)
            copy(image_path, new_path)
            # 实时显示拆分进度
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")
        print()

if __name__ == "__main__":
    # 获取 cell 文件夹下除 .txt 文件以外所有文件夹名（即5种宫颈细胞的类名）
    file_path = "../dataset/SIPaKMeD/cell/"
    base_dir = "../dataset/SIPaKMeD"
    cell_classes = [cla for cla in os.listdir(file_path) if ".txt" not in cla]

    # 创建数据集目录结构
    create_dataset_directories(base_dir, cell_classes)

    # 划分数据集
    split_dataset(file_path, base_dir, cell_classes)

    print("processing done!")
