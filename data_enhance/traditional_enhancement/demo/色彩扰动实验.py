# -*- coding:utf-8 -*-
"""
作者：尘小风
日期：2023年05月13日
软件：Pycharm2020.2
"""

import os
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import ImageEnhance, Image


file_name = "test.bmp"
save_name = "001_01_"

img_original = cv2.imread(file_name)

# 获取颜色通道数
c = img_original.shape[2]
# 通道打乱
def channel_shuffle(img):
    c_shuffle = list(range(c))  # range(c)返回的是一个可迭代的对象，不是列表，需要转换成list格式
    random.shuffle(c_shuffle)  # random.shuffle没有返回值，所以不能写成c_shuffle=random.shuffle(c_shuffle)
    img = img[..., c_shuffle]
    return img

# 颜色通道随机打乱
img_shufflechannels = channel_shuffle(img_original)
save_imgname1 = save_name + "_shufflechannels.bmp"
save_path1 = os.path.join(os.getcwd(), save_imgname1)
cv2.imwrite(save_path1, img_shufflechannels)


def randomColor(image, saturation=0, brightness=0, contrast=0, sharpness=0):
    if random.random() < saturation: # random.random()随机生成[0,1)之间的实数
        random_factor = np.random.randint(20, 31) / 10.  # 随机因子，random.randint随机生成[0,31)之间的整数
        image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    if random.random() < brightness:
        random_factor = np.random.randint(20, 21) / 10.  # 随机因子
        image = ImageEnhance.Brightness(image).enhance(random_factor)  # 调整图像的亮度
    if random.random() < contrast:
        random_factor = np.random.randint(20, 21) / 10.  # 随机因子
        image = ImageEnhance.Contrast(image).enhance(random_factor)  # 调整图像对比度
    if random.random() < sharpness:
        random_factor = np.random.randint(100, 101) / 10.  # 随机因子
        ImageEnhance.Sharpness(image).enhance(random_factor)  # 调整图像锐度
    return image

img = cv2.imread(file_name)
img_original = Image.fromarray(img)

# 调整饱和度
# np.asarray实现image到array的转换，cv2.imwrite需要对图像的矩阵(numpy)进行保存，如果不进行转换会报错
img_color_enhance = np.asarray(randomColor(img_original, saturation=1))
save_imgname2 = save_name + "_color_enhance.bmp"
save_path2 = os.path.join(os.getcwd(), save_imgname2)
cv2.imwrite(save_path2, img_color_enhance)

# 调整亮度
img_bright_enhance = np.asarray(randomColor(img_original, brightness=1))
save_imgname3 = save_name + "_bright_enhance.bmp"
save_path3 = os.path.join(os.getcwd(), save_imgname3)
cv2.imwrite(save_path3, img_bright_enhance)

# 调整对比度
img_contrast_enhance = np.asarray(randomColor(img_original, contrast=1))
save_imgname4 = save_name + "_contrast_enhance.bmp"
save_path4 = os.path.join(os.getcwd(), save_imgname4)
cv2.imwrite(save_path4, img_contrast_enhance)

# 调整锐度
img_sharpness_enhance = np.asarray(randomColor(img_original, sharpness=1))
save_imgname5 = save_name + "_sharpness_enhance.bmp"
save_path5 = os.path.join(os.getcwd(), save_imgname5)
cv2.imwrite(save_path5, img_sharpness_enhance)


plt.subplot(231)
plt.imshow(img[:,:,::-1])
plt.title("original")

plt.subplot(232)
plt.imshow(img_shufflechannels[:,:,::-1])
plt.title("channel_shuffle")

plt.subplot(233)
plt.imshow(img_color_enhance[:,:,::-1])
plt.title("color_enhance")

plt.subplot(234)
plt.imshow(img_bright_enhance[:,:,::-1])
plt.title("bright_enhance")

plt.subplot(235)
plt.imshow(img_contrast_enhance[:,:,::-1])
plt.title("contrast_enhance")

plt.subplot(236)
plt.imshow(img_sharpness_enhance[:,:,::-1])
plt.title("sharpness_enhance")


plt.show()
