# -*- coding:utf-8 -*-
"""
作者：尘小风
软件：Pycharm2020.2
"""

import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt


# 添加椒盐噪声
# 椒盐噪声:一种随机出现的白点或者黑点，可能是亮的区域有黑色像素或是在暗的区域有白色像素（或是两者皆有）
def sp_noise(image):
      output = np.zeros(image.shape, np.uint8)
      prob = np.random.uniform(0.01, 0.05)  #随机噪声比例，随机生成0.01到0.05之间的浮点数，包括两个边界值
      thres = 1 - prob
      for i in range(image.shape[0]): # i图像的宽度w
          for j in range(image.shape[1]): # 图像的高度h
              rdn = np.random.random() # 随机生成一个0到1之间的浮点数，包括0
              if rdn < prob:
                output[i][j] = 0 # 随机添加白点
              elif rdn > thres:
                output[i][j] = 255 # 随机添加黑点
              else:
                output[i][j] = image[i][j] # 图像正常区域
      return output


# 添加高斯噪声
# 高斯噪声：概率密度函数服从高斯分布; mean:均值, var:方差
def gasuss_noise(image, mean, var):

    image = np.array(image / 255, dtype=float) # 图像灰度标准化
    noise = np.random.normal(mean, var ** 0.5, image.shape) # 正态分布
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    '''
    numpy.clip(a, a_min, a_max)
    a: 输入的数组
    a_min: 限定的最小值，也可以是数组，如果为数组时，shape必须和a一样
    a_max: 限定的最大值，也可以是数组，shape和a一样
    '''
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


file_name = "../test.bmp"

folder_name = "result_imgs"
os.makedirs(folder_name, exist_ok=True)
save_root = folder_name + "/"
save_name = os.path.join(save_root, "generate_")

img_original = cv2.imread(file_name)

# 添加高斯噪声
img_gasuss = gasuss_noise(img_original, 0, 0.005)
img_name_1 = save_name + "gususs.bmp"
save_path_1 = os.path.join(os.getcwd(), img_name_1)
cv2.imwrite(save_path_1, img_gasuss)

# 添加椒盐噪声
img_sp = sp_noise(img_original)
img_name_2 = save_name + "sp.bmp"
save_path_2 = os.path.join(os.getcwd(), img_name_2)
cv2.imwrite(save_path_2, img_sp)
