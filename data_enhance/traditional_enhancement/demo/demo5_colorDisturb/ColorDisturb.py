# -*- coding:utf-8 -*-
"""
作者：尘小风
软件：Pycharm2020.2
"""

import os
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import ImageEnhance, Image

def read_image(file_path):
    """
    读取图像文件并进行错误处理

    :param file_path: 图像文件的路径
    :return: 读取到的图像，若读取失败则返回 None
    """
    img = cv2.imread(file_path)
    if img is None:
        print(f"无法读取图像文件: {file_path}")
    return img


def create_save_folder(folder_name):
    """
    创建保存结果的文件夹

    :param folder_name: 文件夹名称
    :return: 保存文件的基础名称
    """
    os.makedirs(folder_name, exist_ok=True)
    save_root = folder_name
    return os.path.join(save_root, "generate_")


def shuffle_channels(img):
    """
    随机打乱图像的颜色通道

    :param img: 输入的图像
    :return: 通道打乱后的图像
    """
    c = img.shape[2]
    c_shuffle = list(range(c))
    random.shuffle(c_shuffle)
    return img[..., c_shuffle]


def enhance_image(image, saturation=0, brightness=0, contrast=0, sharpness=0):
    """
    随机调整图像的饱和度、亮度、对比度和锐度

    :param image: 输入的 PIL 图像
    :param saturation: 调整饱和度的概率
    :param brightness: 调整亮度的概率
    :param contrast: 调整对比度的概率
    :param sharpness: 调整锐度的概率
    :return: 调整后的 PIL 图像
    """
    if random.random() < saturation:
        random_factor = np.random.uniform(2.0, 3.1)
        image = ImageEnhance.Color(image).enhance(random_factor)
    if random.random() < brightness:
        random_factor = np.random.uniform(2.0, 2.1)
        image = ImageEnhance.Brightness(image).enhance(random_factor)
    if random.random() < contrast:
        random_factor = np.random.uniform(2.0, 2.1)
        image = ImageEnhance.Contrast(image).enhance(random_factor)
    if random.random() < sharpness:
        random_factor = np.random.uniform(10.0, 10.1)
        image = ImageEnhance.Sharpness(image).enhance(random_factor)
    return image


def save_image(img, save_name, suffix):
    """
    保存处理后的图像

    :param img: 要保存的图像
    :param save_name: 保存文件的基础名称
    :param suffix: 文件名后缀
    :return: 保存的图像路径
    """
    save_imgname = save_name + suffix + ".bmp"
    save_path = os.path.join(os.getcwd(), save_imgname)
    cv2.imwrite(save_path, img)
    return save_path


def main():
    file_name = "../test.bmp"
    folder_name = "result_imgs"

    save_name = create_save_folder(folder_name)
    img_original = read_image(file_name)
    if img_original is None:
        return

    # 颜色通道随机打乱
    img_shufflechannels = shuffle_channels(img_original)
    save_image(img_shufflechannels, save_name, "_shufflechannels")

    img_pil = Image.fromarray(img_original)

    enhancements = [
        ("_color_enhance", 1, 0, 0, 0),
        ("_bright_enhance", 0, 1, 0, 0),
        ("_contrast_enhance", 0, 0, 1, 0),
        ("_sharpness_enhance", 0, 0, 0, 1)
    ]

    enhanced_images = [img_original, img_shufflechannels]

    for suffix, sat, bright, cont, sharp in enhancements:
        enhanced_img = enhance_image(img_pil, sat, bright, cont, sharp)
        img_np = np.asarray(enhanced_img)
        save_image(img_np, save_name, suffix)
        enhanced_images.append(img_np)

    # 显示图像
    titles = [
        "original", "channel_shuffle", "color_enhance",
        "bright_enhance", "contrast_enhance", "sharpness_enhance"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(len(enhanced_images)):
        axes[i].imshow(enhanced_images[i][:, :, ::-1])
        axes[i].set_title(titles[i])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()