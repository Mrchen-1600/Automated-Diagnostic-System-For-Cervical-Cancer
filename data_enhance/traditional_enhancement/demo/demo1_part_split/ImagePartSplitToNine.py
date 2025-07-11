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

import os
from PIL import Image

file_name = "../test.bmp"
folder_name = "result_imgs"

# 创建保存结果的文件夹
os.makedirs(folder_name, exist_ok=True)
save_root = os.path.join(folder_name)
save_name_base = os.path.join(save_root, "generate_")

# 打开图片并设置最大像素限制
img_original = Image.open(file_name)
Image.MAX_IMAGE_PIXELS = None
m, n = img_original.size  # 读取图片的宽度和高度
w = m // 3  # 设置要裁剪的小图的宽度，使用整除操作得到整数
h = n // 3  # 设置要裁剪的小图的高度，使用整除操作得到整数

# 定义裁剪区域
crop_areas = [
    (0, 0, w, h),
    (w, 0, 2 * w, h),
    (2 * w, 0, m, h),
    (0, h, w, 2 * h),
    (w, h, 2 * w, 2 * h),
    (2 * w, h, m, 2 * h),
    (0, 2 * h, w, n),
    (w, 2 * h, 2 * w, n),
    (2 * w, 2 * h, m, n)
]

# 使用循环进行裁剪和保存操作
for i, area in enumerate(crop_areas, start=1):
    region = img_original.crop(tuple(map(int, area)))  # 裁剪区域并确保参数为整数
    save_path = f"{save_name_base}{i}.jpg"
    region.save(save_path)