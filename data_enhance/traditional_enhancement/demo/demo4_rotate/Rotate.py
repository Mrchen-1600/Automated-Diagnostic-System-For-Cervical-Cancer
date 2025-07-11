# -*- coding:utf-8 -*-
"""
作者：尘小风
软件：Pycharm2020.2
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# 定义裁剪函数
def crop_image(img, x0, y0, w, h):
    """
    裁剪图像的指定区域

    :param img: 输入的图像
    :param x0: 裁剪区域的左上角 x 坐标
    :param y0: 裁剪区域的左上角 y 坐标
    :param w: 裁剪区域的宽度
    :param h: 裁剪区域的高度
    :return: 裁剪后的图像
    """
    return img[y0:y0 + h, x0:x0 + w]


def rotate_image(img, angle, crop):
    """
    对输入图像进行指定角度的旋转，并可选择是否裁剪黑边

    :param img: 输入的图像
    :param angle: 旋转的角度
    :param crop: 是否需要进行裁剪，布尔值
    :return: 旋转后的图像
    """
    h, w = img.shape[:2]  # 修正获取宽高的顺序
    # 旋转角度的周期是360°
    angle %= 360
    # 计算仿射变换矩阵
    M_rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    # 得到旋转后的图像
    img_rotated = cv2.warpAffine(img, M_rotation, (w, h))

    # 如果需要去除黑边
    if crop:
        # 裁剪角度的等效周期是180°
        angle_crop = angle % 180
        if angle > 90:
            angle_crop = 180 - angle_crop
        # 转化角度为弧度
        theta = np.deg2rad(angle_crop)
        # 计算高宽比
        hw_ratio = h / w
        # 计算裁剪边长系数的分子项
        tan_theta = np.abs(np.tan(theta))
        numerator = np.abs(np.cos(theta) + np.sin(theta) * tan_theta)
        # 计算分母中和高宽比相关的项
        r = hw_ratio if h > w else 1 / hw_ratio
        # 计算分母项
        denominator = r * tan_theta + 1
        # 最终的边长系数
        crop_mult = numerator / denominator

        # 得到裁剪区域
        w_crop = int(crop_mult * w)
        h_crop = int(crop_mult * h)
        x0 = (w - w_crop) // 2
        y0 = (h - h_crop) // 2

        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)

    return img_rotated


def save_rotated_image(img, angle, save_name):
    """
    旋转图像并保存

    :param img: 输入的图像
    :param angle: 旋转的角度
    :param save_name: 保存的文件名前缀
    :return: 旋转后的图像
    """
    rotated_img = rotate_image(img, angle, False)
    save_imgname = f"{save_name}angle{angle}.bmp"
    save_path = os.path.join(os.getcwd(), save_imgname)
    cv2.imwrite(save_path, rotated_img)
    return rotated_img

def main():
    file_name = "../test.bmp"
    folder_name = "result_imgs"
    # 创建保存结果的文件夹
    os.makedirs(folder_name, exist_ok=True)
    save_root = os.path.join(folder_name)
    save_name = os.path.join(save_root, "generate_")

    # 读取图像并进行错误处理
    img_original = cv2.imread(file_name)
    if img_original is None:
        print(f"无法读取图像文件: {file_name}")
        return

    angles = [45, 90, 135, 180, -45, -90, -135]
    rotated_images = [img_original]

    # 旋转并保存图像
    for angle in angles:
        rotated_img = save_rotated_image(img_original, angle, save_name)
        rotated_images.append(rotated_img)

    # 显示图像
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    for i, img in enumerate(rotated_images[:9]):
        axes[i].imshow(img[:, :, ::-1])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()