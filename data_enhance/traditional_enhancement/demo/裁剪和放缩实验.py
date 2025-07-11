# -*- coding:utf-8 -*-
"""
作者：尘小风
日期：2023年05月12日
软件：Pycharm2020.2
"""

import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import os
from PIL import Image

file_name = "../test.bmp"
save_name = "001_01_"


img_original = Image.open(file_name)


# 放大
img_max = transforms.Resize(140)
centor_crop = transforms.CenterCrop((112, 109))
img_enlarge = centor_crop(img_max(img_original))
save_imgname1 = save_name + "enlarge.bmp"
save_path1 = os.path.join(os.getcwd(), save_name + save_imgname1)
img_enlarge.save(save_path1)

# 随机裁剪
random_crop = transforms.RandomResizedCrop(90)
img_randomcrop = random_crop(img_original)
resize = transforms.Resize((112, 109))
img_resize = resize(img_randomcrop)
save_imgname2 = save_name + "randomcrop.bmp"
save_path2 = os.path.join(os.getcwd(), save_name + save_imgname2)
img_resize.save(save_path2)


