# -*- coding:utf-8 -*-
"""
作者：尘小风
软件：Pycharm2020.2
"""
import os
import cv2
import numpy as np

# 添加椒盐噪声
# 椒盐噪声:一种随机出现的白点或者黑点，可能是亮的区域有黑色像素或是在暗的区域有白色像素（或是两者皆有）
def sp_noise(image):
    prob = np.random.uniform(0.01, 0.05)  # 随机噪声比例，随机生成0.01到0.05之间的浮点数
    output = image.copy()
    # 生成随机掩码
    mask = np.random.random(image.shape[:2])
    output[mask < prob] = 0  # 随机添加黑点
    output[mask > 1 - prob] = 255  # 随机添加白点
    return output

# 添加高斯噪声
# 高斯噪声：概率密度函数服从高斯分布; mean:均值, var:方差
def gaussian_noise(image, mean, var):
    # 图像灰度标准化
    image = image.astype(np.float32) / 255.0
    # 生成高斯噪声
    noise = np.random.normal(mean, np.sqrt(var), image.shape)
    out = image + noise
    # 裁剪数值范围
    out = np.clip(out, 0.0, 1.0)
    # 恢复图像数据类型
    out = (out * 255).astype(np.uint8)
    return out

file_name = "../test.bmp"
folder_name = "result_imgs"

# 创建保存文件夹
os.makedirs(folder_name, exist_ok=True)
save_root = os.path.join(folder_name)
save_name_base = os.path.join(save_root, "generate_")

# 读取图像
img_original = cv2.imread(file_name)
if img_original is None:
    print(f"无法读取图像文件: {file_name}")
else:
    # 添加高斯噪声
    img_gaussian = gaussian_noise(img_original, 0, 0.005)
    img_name_1 = save_name_base + "gaussian.bmp"
    save_path_1 = os.path.join(os.getcwd(), img_name_1)
    cv2.imwrite(save_path_1, img_gaussian)

    # 添加椒盐噪声
    img_sp = sp_noise(img_original)
    img_name_2 = save_name_base + "sp.bmp"
    save_path_2 = os.path.join(os.getcwd(), img_name_2)
    cv2.imwrite(save_path_2, img_sp)