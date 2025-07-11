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


file_name = "../test.bmp"

folder_name = "result_imgs"
os.makedirs(folder_name, exist_ok=True)
save_root = folder_name + "/"
save_name = os.path.join(save_root, "generate_")

img_original = cv2.imread(file_name)

# 滤波处理（滤波处理主要是让图像变得模糊，提取图像的重要信息）
img1 = cv2.blur(img_original, (5, 5))  # 均值滤波
img_name1 = save_name + "blur.bmp"
save_path1 = os.path.join(os.getcwd(), img_name1)
cv2.imwrite(save_path1, img1)

img2 = cv2.medianBlur(img_original, 9)  # 中值滤波
img_name2 = save_name + "medianBlur.bmp"
save_path2 = os.path.join(os.getcwd(), img_name2)
cv2.imwrite(save_path2, img2)

img3 = cv2.GaussianBlur(img_original, (15, 15), 0)  # 高斯滤波
img_name3 = save_name + "GaussianBlur.bmp"
save_path3 = os.path.join(os.getcwd(), img_name3)
cv2.imwrite(save_path3, img3)


seq4 = iaa.Sequential([ #建立一个名为seq的实例，定义增强方法，用于增强
    # 将1%到5%的像素设置为黑色, 对50%的图像每个颜色通道单独进行此操作（每个通道都随机将1%到5%的像素设置为黑色）
    iaa.CoarseDropout((0.05, 0.10), size_percent=(0.05, 0.1), per_channel=0.5)])

seq5 = iaa.Sequential([
    # 将3%到10%的像素用原图大小2%到5%的黑色方块覆盖，per_channel=0.2对20%的图像每个颜色通道单独进行此操作
    iaa.CoarseDropout(
        (0.05, 0.10), size_percent=(0.05, 0.1))])


# # 将1%到5%的像素设置为黑色
img_dropout = seq4.augment_image(img_original)
img_name4 = save_name + "dropout.bmp"
save_path4 = os.path.join(os.getcwd(), img_name4)
cv2.imwrite(save_path4, img_dropout)


# 将3%到10%的像素用原图大小2%到5%的黑色方块覆盖
img_coarsedropout = seq5.augment_image(img_original)
img_name5 = save_name + "coarse_dropout.bmp"
save_path5 = os.path.join(os.getcwd(), img_name5)
cv2.imwrite(save_path5, img_coarsedropout)

# 因为cv2.imread()函数返回的图像颜色空间是BGR，而不是RGB
# 所以显示图片的时候可以通过[:,:,::-1]让图像颜色空间倒序排列，即BGR->RGB
plt.subplot(231)
plt.imshow(img_original[:,:,::-1])

plt.subplot(232)
plt.imshow(img1[:,:,::-1])

plt.subplot(233)
plt.imshow(img2[:,:,::-1])

plt.subplot(234)
plt.imshow(img3[:,:,::-1])

plt.subplot(235)
plt.imshow(img_dropout[:,:,::-1])

plt.subplot(236)
plt.imshow(img_coarsedropout[:,:,::-1])

plt.show()