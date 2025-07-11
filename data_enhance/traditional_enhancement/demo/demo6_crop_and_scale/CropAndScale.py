# -*- coding:utf-8 -*-
"""
作者：尘小风
软件：Pycharm2020.2
"""

import os
from PIL import Image
from torchvision import transforms

def create_save_folder(folder_name):
    """
    创建保存结果的文件夹，并返回保存文件的基础路径

    :param folder_name: 文件夹名称
    :return: 保存文件的基础路径
    """
    os.makedirs(folder_name, exist_ok=True)
    return os.path.join(folder_name, "generate_")


def load_image(file_path):
    """
    加载图像文件并进行错误处理

    :param file_path: 图像文件路径
    :return: 加载的图像对象，若加载失败则返回 None
    """
    try:
        return Image.open(file_path)
    except FileNotFoundError:
        print(f"错误：未找到图像文件 {file_path}")
        return None
    except Exception as e:
        print(f"加载图像时出错 {file_path}: {e}")
        return None


def save_image(img, save_base_name, suffix):
    """
    保存处理后的图像

    :param img: 要保存的图像对象
    :param save_base_name: 保存文件的基础路径
    :param suffix: 文件名后缀
    :return: 保存的图像路径
    """
    save_imgname = save_base_name + suffix + ".bmp"
    img.save(save_imgname)
    return save_imgname


def process_image(img, transform_list, save_base_name, suffix):
    """
    对图像进行一系列变换并保存

    :param img: 输入的图像对象
    :param transform_list: 变换操作列表
    :param save_base_name: 保存文件的基础路径
    :param suffix: 文件名后缀
    :return: 处理并保存后的图像路径
    """
    processed_img = img
    for transform in transform_list:
        processed_img = transform(processed_img)
    return save_image(processed_img, save_base_name, suffix)


def main():
    file_name = "../test.bmp"
    folder_name = "result_imgs"

    save_base_name = create_save_folder(folder_name)
    img_original = load_image(file_name)
    if img_original is None:
        return

    # 放大操作
    enlarge_transforms = [
        transforms.Resize(140),
        transforms.CenterCrop((112, 109))
    ]
    process_image(img_original, enlarge_transforms, save_base_name, "enlarge")

    # 随机裁剪操作
    random_crop_transforms = [
        transforms.RandomResizedCrop(90),
        transforms.Resize((112, 109))
    ]
    process_image(img_original, random_crop_transforms, save_base_name, "randomcrop")

if __name__ == "__main__":
    main()