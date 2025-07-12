# -*- coding= utf-8 -*-

# @Author : 尘小风
# @File : RemoveBlackEdge.py
# @software : PyCharm

import os
import cv2
import numpy as np

def remove_images_with_excessive_black_pixels(file_root, threshold=0.3):
    """
    移除黑色像素比例超过阈值的图片，会递归处理所有子文件夹

    :param file_root: 图片所在文件夹路径
    :param threshold: 黑色像素比例阈值，默认为 0.3
    """
    try:
        count = 0
        # 使用 os.walk 递归遍历所有子文件夹
        for root, dirs, files in os.walk(file_root):
            for img_wholename in files:
                img_path = os.path.join(root, img_wholename)
                try:
                    # 读取图像
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"警告：无法读取图像 {img_path}，跳过该文件。")
                        continue

                    # 获取图像高度和宽度
                    height, width = img.shape[:2]
                    total_pixels = width * height

                    count = count + 1

                    # 使用 NumPy 向量化操作统计黑色像素数量
                    black_pixels = np.sum(np.all(img == [0, 0, 0], axis=-1))

                    # 计算黑色像素比例
                    black_ratio = black_pixels / total_pixels

                    if black_ratio > threshold:
                        os.remove(img_path)
                        print(f"已移除图像 {img_path}，黑色像素比例: {black_ratio:.2f}")
                except Exception as e:
                    print(f"处理图像 {img_path} 时出错: {e}")

        print(f"共处理 {count} 张图片")
    except FileNotFoundError:
        print(f"错误：文件夹 {file_root} 未找到。")
        return

# 图片所在文件夹路径
file_root = "../dataset/SIPaKMeD/train"
# 调用函数移除黑色像素比例超过阈值的图片
remove_images_with_excessive_black_pixels(file_root)