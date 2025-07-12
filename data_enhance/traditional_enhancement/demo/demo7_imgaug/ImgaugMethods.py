# -*- coding= utf-8 -*-
# @Author : 尘小风
# @File : ImgaugMethods.py
# @software : PyCharm

import os
import imgaug as ia
import cv2
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa


def sometimes(aug):
    """
    对 50% 的图像应用指定的增强操作

    :param aug: 要应用的增强操作
    :return: 封装后的增强操作，该操作会以 50% 的概率作用于图像
    """
    return iaa.Sometimes(0.5, aug)

# 定义各种图像增强操作的参数，方便后续修改和管理
# 水平翻转概率，50% 的图像会进行水平翻转
FLIPLR_PROB = 0.5
# 垂直翻转概率，20% 的图像会进行垂直翻转
FLIPUD_PROB = 0.2
# 裁剪比例范围，图像会随机裁剪 0 到 10% 的区域
CROP_PERCENT = (0, 0.1)
# 仿射变换的参数
AFFINE_PARAMS = {
    # 图像在 x 和 y 方向的缩放比例范围，为 80% 到 120% 之间
    "scale": {"x": (0.8, 1.2), "y": (0.8, 1.2)},
    # 图像在 x 和 y 方向的平移比例范围，为 ±20% 之间
    "translate_percent": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    # 图像旋转角度范围，为 ±45 度之间
    "rotate": (-45, 45),
    # 图像剪切变换角度范围，为 ±16 度之间
    "shear": (-16, 16),
    # 插值方法，使用最邻近插值或者双线性插值其中一种
    "order": [0, 1],
    # 填充值范围，用 0 到 255 之间的数填充变换后的空白区域
    "cval": (0, 255),
    # 定义填充图像外区域的方法，ia.ALL 表示从几种方法中任选一种
    "mode": ia.ALL
}

# 超像素处理的参数
SUPERPIXELS_PARAMS = {
    # 用超像素替换原始像素的概率范围
    "p_replace": (0, 1.0),
    # 超像素的分割数量范围
    "n_segments": (20, 200)
}

# 边缘检测的参数
EDGE_DETECT_PARAMS = {
    # 边缘检测结果与原图叠加的透明度范围
    "alpha": (0, 0.7)
}

# 定向边缘检测的参数
DIRECTED_EDGE_DETECT_PARAMS = {
    # 定向边缘检测结果与原图叠加的透明度范围
    "alpha": (0, 0.7),
    # 边缘检测的方向范围
    "direction": (0.0, 1.0)
}

# 弹性变换的参数
ELASTIC_TRANSFORM_PARAMS = {
    # 弹性变换的强度范围
    "alpha": (0.5, 3.5),
    # 弹性变换的平滑度
    "sigma": 0.25
}

# 分段仿射变换的参数
PIECEWISE_AFFINE_PARAMS = {
    # 分段仿射变换的缩放比例范围
    "scale": (0.01, 0.05)
}

# 构建增强序列，将各种增强操作组合在一起
augmentations = [
    # 对 50% 的图像进行水平翻转
    iaa.Fliplr(FLIPLR_PROB),
    # 对 20% 的图像进行垂直翻转
    iaa.Flipud(FLIPUD_PROB),
    # 以 50% 的概率对图像进行裁剪操作
    sometimes(iaa.Crop(percent=CROP_PERCENT)),
    # 以 50% 的概率对图像进行仿射变换
    sometimes(iaa.Affine(**AFFINE_PARAMS)),
    # 从列表中随机选择 0 到 5 个增强操作应用到图像上
    iaa.SomeOf((0, 5), [
        # 以 50% 的概率对图像进行超像素处理
        sometimes(iaa.Superpixels(**SUPERPIXELS_PARAMS)),
        # 从三种模糊操作中随机选择一种应用到图像上
        iaa.OneOf([
            # 高斯模糊，模糊程度为 0 到 3.0 之间
            iaa.GaussianBlur((0, 3.0)),
            # 均值模糊，核大小为 2 到 7 之间
            iaa.AverageBlur(k=(2, 7)),
            # 中值模糊，核大小为 3 到 11 之间
            iaa.MedianBlur(k=(3, 11))
        ]),
        # 对图像进行锐化处理，锐化强度和亮度调整范围可配置
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
        # 对图像进行浮雕效果处理，效果强度可配置
        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
        # 以 50% 的概率从边缘检测和定向边缘检测中选择一种应用到图像上
        sometimes(iaa.OneOf([
            iaa.EdgeDetect(**EDGE_DETECT_PARAMS),
            iaa.DirectedEdgeDetect(**DIRECTED_EDGE_DETECT_PARAMS)
        ])),
        # 给图像添加高斯噪声，噪声强度和通道应用概率可配置
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # 从像素丢弃和粗粒度像素丢弃中选择一种应用到图像上
        iaa.OneOf([
            # 随机将 1% 到 10% 的像素设置为黑色
            iaa.Dropout((0.01, 0.1), per_channel=0.5),
            # 随机用黑色方块覆盖 3% 到 15% 的像素，方块大小为原图的 2% 到 5%
            iaa.CoarseDropout(
                (0.03, 0.15), size_percent=(0.02, 0.05),
                per_channel=0.2
            )
        ]),
        # 以 5% 的概率反转图像像素的强度
        iaa.Invert(0.05, per_channel=True),
        # 给每个像素随机加减 -10 到 10 之间的数，部分通道应用
        iaa.Add((-10, 10), per_channel=0.5),
        # 让每个像素乘上 0.5 到 1.5 之间的数字，部分通道应用
        iaa.Multiply((0.5, 1.5), per_channel=0.5),
        # 将整个图像的对比度变为原来的 0.5 到 2 倍，部分通道应用
        iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
        # 将图像转换为灰度图后按比例叠加到原图上
        iaa.Grayscale(alpha=(0.0, 1.0)),
        # 以 50% 的概率对图像进行弹性变换
        sometimes(iaa.ElasticTransformation(**ELASTIC_TRANSFORM_PARAMS)),
        # 以 50% 的概率对图像进行分段仿射变换
        sometimes(iaa.PiecewiseAffine(**PIECEWISE_AFFINE_PARAMS))
    ], random_order=True)
]

# 创建一个增强序列对象，随机顺序应用增强操作
seq = iaa.Sequential(augmentations, random_order=True)


def read_image(file_path):
    """
    读取图像

    :param file_path: 图像文件路径
    :return: 读取的图像，若读取失败则返回 None
    """
    img = cv2.imread(file_path)
    if img is None:
        print(f"无法读取图像文件: {file_path}")
    return img


def save_image(img, file_path):
    """
    保存图像

    :param img: 要保存的图像
    :param file_path: 保存的文件路径
    :return: 保存成功返回 True，失败返回 False
    """
    try:
        cv2.imwrite(file_path, img)
        return True
    except Exception as e:
        print(f"保存图像时出错 {file_path}: {e}")
        return False


def show_image(img):
    """
    显示图像

    :param img: 要显示的图像
    """
    # 由于 cv2.imread() 返回的图像是 BGR 格式，而 matplotlib 显示需要 RGB 格式，因此进行通道反转
    plt.imshow(img[:, :, ::-1])
    plt.show()

def main():
    input_image_path = "../test.bmp"

    folder_name = "result_imgs"
    # 创建保存结果的文件夹
    os.makedirs(folder_name, exist_ok=True)
    save_root = os.path.join(folder_name)
    output_image_path = os.path.join(save_root, "generate_")


    # 读取输入图像
    img = read_image(input_image_path)

    for i in range(10):
        # 对图像应用增强序列进行图像增强
        images_aug = seq.augment_image(img)
        # 保存增强后的图像
        if save_image(images_aug, output_image_path + str(i) + ".jpg"):
            # 若图像保存成功，显示增强后的图像
            show_image(images_aug)

if __name__ == "__main__":
    # 当脚本作为主程序运行时，调用 main 函数
    main()