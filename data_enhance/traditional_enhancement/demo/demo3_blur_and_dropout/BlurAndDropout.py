# -*- coding:utf-8 -*-
"""
作者：尘小风
日期：2025年07月10日
软件：Pycharm2020.2
"""

import os
import cv2
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

def apply_filter(image, filter_func, *args, **kwargs):
    """
    应用指定的滤波函数处理图像

    :param image: 输入图像
    :param filter_func: 滤波函数
    :param args: 滤波函数的位置参数
    :param kwargs: 滤波函数的关键字参数
    :return: 处理后的图像
    """
    return filter_func(image, *args, **kwargs)


def save_image(image, save_path):
    """
    保存图像到指定路径

    :param image: 要保存的图像
    :param save_path: 保存路径
    """
    cv2.imwrite(save_path, image)


def show_images(images, titles):
    """
    显示多个图像

    :param images: 图像列表
    :param titles: 图像标题列表
    """
    rows = 2
    cols = 3
    for i in range(len(images)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i][:, :, ::-1])
        plt.title(titles[i])
        plt.axis('off')
    plt.show()


file_name = "../test.bmp"
folder_name = "result_imgs"

# 创建保存结果的文件夹
os.makedirs(folder_name, exist_ok=True)
save_root = os.path.join(folder_name)
save_name_base = os.path.join(save_root, "generate_")

# 读取图像
img_original = cv2.imread(file_name)
if img_original is None:
    print(f"无法读取图像文件: {file_name}")
else:
    filters = [
        (cv2.blur, (5, 5), "blur"),
        (cv2.medianBlur, 9, "medianBlur"),
        (cv2.GaussianBlur, (15, 15), 0, "GaussianBlur")
    ]

    augmented_images = []
    image_titles = ["Original"]
    processed_images = [img_original]

    # 应用滤波处理
    for filter_info in filters:
        filter_func = filter_info[0]
        args = filter_info[1:-1]
        filter_name = filter_info[-1]
        img_filtered = apply_filter(img_original, filter_func, *args)
        img_name = f"{save_name_base}{filter_name}.bmp"
        save_path = os.path.join(os.getcwd(), img_name)
        save_image(img_filtered, save_path)
        processed_images.append(img_filtered)
        image_titles.append(filter_name)

    # 定义图像增强序列
    seq4 = iaa.Sequential([
        iaa.CoarseDropout((0.05, 0.10), size_percent=(0.05, 0.1), per_channel=0.5)
    ])

    seq5 = iaa.Sequential([
        iaa.CoarseDropout((0.05, 0.10), size_percent=(0.05, 0.1))
    ])

    # 应用图像增强
    augmentations = [
        (seq4, "dropout"),
        (seq5, "coarse_dropout")
    ]

    for seq, aug_name in augmentations:
        img_augmented = seq.augment_image(img_original)
        img_name = f"{save_name_base}{aug_name}.bmp"
        save_path = os.path.join(os.getcwd(), img_name)
        save_image(img_augmented, save_path)
        processed_images.append(img_augmented)
        image_titles.append(aug_name)

    # 显示处理后的图像
    show_images(processed_images, image_titles)