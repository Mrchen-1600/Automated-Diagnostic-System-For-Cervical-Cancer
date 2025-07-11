# -*- coding:utf-8 -*-
"""
作者：尘小风
软件：Pycharm2020.2
"""

# -*- coding:utf-8 -*-
"""
作者：尘小风
软件：Pycharm2020.2
"""

from PIL import Image
import os

file_name = "../test.bmp"

folder_name = "result_imgs"
os.makedirs(folder_name, exist_ok=True)
save_root = folder_name + "/"
save_name = os.path.join(save_root, "generate_")

img_original = Image.open(file_name)
Image.MAX_IMAGE_PIXELS = None
img_size = img_original.size
m = img_size[0]  # 读取图片的宽度
n = img_size[1]  # 读取图片的高度
w = m / 3  # 设置要裁剪的小图的宽度
h = n / 3  # 设置要裁剪的小图的高度

region_1 = img_original.crop((0, 0, w, h))  # 裁剪区域
region_1.save(save_name + "1.jpg")

region_2 = img_original.crop((w, 0, 2 * w, h))  # 裁剪区域
region_2.save(save_name + "2.jpg")

region_3 = img_original.crop((2 * w, 0, m, h))  # 裁剪区域
region_3.save(save_name + "3.jpg")

region_4 = img_original.crop((0, h, w, 2 * h))  # 裁剪区域
region_4.save(save_name + "4.jpg")

region_5 = img_original.crop((w, h, 2 * w, 2 * h))  # 裁剪区域
region_5.save(save_name + "5.jpg")

region_6 = img_original.crop((2*w, h, m, 2 * h))  # 裁剪区域
region_6.save(save_name + "6.jpg")

region_7 = img_original.crop((0, 2 * h, w, n))  # 裁剪区域
region_7.save(save_name + "7.jpg")

region_8 = img_original.crop((w, 2 * h, 2 * w, n))  # 裁剪区域
region_8.save(save_name + "8.jpg")

region_9 = img_original.crop((2*w, 2 * h, m, n))  # 裁剪区域
region_9.save(save_name + "9.jpg")