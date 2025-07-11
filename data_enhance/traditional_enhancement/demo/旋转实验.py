# -*- coding:utf-8 -*-
"""
作者：尘小风
日期：2023年05月12日
软件：Pycharm2020.2
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# 定义任意角度旋转函数，黑边被裁剪了，图片原尺寸会改变
crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]
def rotate_image(img, angle, crop):
    """
    img: 输入的图像
    angle: 旋转的角度
    crop: 是否需要进行裁剪，布尔向量
    """
    w, h = img.shape[:2]
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
        theta = angle_crop * np.pi / 180
        # 计算高宽比
        hw_ratio = float(h) / float(w)
        # 计算裁剪边长系数的分子项
        tan_theta = abs(np.tan(theta))
        numerator = abs(np.cos(theta) + np.sin(theta) * np.tan(theta))

        # 计算分母中和高宽比相关的项
        r = hw_ratio if h > w else 1 / hw_ratio
        # 计算分母项
        denominator = r * tan_theta + 1
        # 最终的边长系数
        crop_mult = numerator / denominator

        # 得到裁剪区域
        w_crop = int(crop_mult * w)
        h_crop = int(crop_mult * h)
        x0 = int((w - w_crop) / 2)
        y0 = int((h - h_crop) / 2)

        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)

    return img_rotated

file_name = "test.bmp"
save_name = "001_01_"


img_original = cv2.imread(file_name)

# 旋转45度
# 第一个传入参数：要处理的图片；第二个传入参数：旋转角度；第三个传入参数：是否需要裁剪黑边
angle45 = rotate_image(img_original, 45, False)
save_imgname1 = save_name + "angle45.bmp"
save_path1 = os.path.join(os.getcwd(), save_name + save_imgname1)
cv2.imwrite(save_path1, angle45)

# 旋转90度
angle90 = rotate_image(img_original, 90, False)
save_imgname1 = save_name + "angle90.bmp"
save_path1 = os.path.join(os.getcwd(), save_name + save_imgname1)
cv2.imwrite(save_path1, angle90)


angle135 = rotate_image(img_original, 135, False)
save_imgname2 = save_name + "angle135.bmp"
save_path2 = os.path.join(os.getcwd(), save_name + save_imgname2)
cv2.imwrite(save_path2, angle135)


# 旋转180度
angle180 = rotate_image(img_original, 180, False)
save_imgname3 = save_name + "angle180.bmp"
save_path3 = os.path.join(os.getcwd(), save_name + save_imgname3)
cv2.imwrite(save_path3, angle180)


# 旋转180度
angle225 = rotate_image(img_original, -45, False)
save_imgname4 = save_name + "angle225.bmp"
save_path4 = os.path.join(os.getcwd(), save_name + save_imgname4)
cv2.imwrite(save_path4, angle225)

# 旋转270度
angle270 = rotate_image(img_original, -90, False)
save_imgname5 = save_name + "angle270.bmp"
save_path5 = os.path.join(os.getcwd(), save_name + save_imgname5)
cv2.imwrite(save_path5, angle270)

angle315 = rotate_image(img_original, -135, False)
save_imgname6 = save_name + "angle315.bmp"
save_path6 = os.path.join(os.getcwd(), save_name + save_imgname6)
cv2.imwrite(save_path6, angle315)

# 因为cv2.imread()函数返回的图像颜色空间是BGR，而不是RGB
# 所以显示图片的时候可以通过[:,:,::-1]让图像颜色空间倒序排列，即BGR->RGB
plt.subplot(331)
plt.imshow(img_original[:,:,::-1])

plt.subplot(332)
plt.imshow(angle45[:,:,::-1])

plt.subplot(333)
plt.imshow(angle90[:,:,::-1])


plt.subplot(334)
plt.imshow(angle135[:,:,::-1])

plt.subplot(335)
plt.imshow(angle180[:,:,::-1])

plt.subplot(336)
plt.imshow(angle225[:,:,::-1])

plt.subplot(337)
plt.imshow(angle270[:,:,::-1])

plt.subplot(338)
plt.imshow(angle315[:,:,::-1])


plt.show()