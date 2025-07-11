# -*- coding= utf-8 -*-
# @Author : 尘小风
# @File : RenameImage.py
# @software : PyCharm

"""
用于给测试集图片重命名，在细胞图像的文件名上加上其正确的类别名，便于网络预测时统计预测的精度
"""

import os
from PIL import Image

# 细胞类别
classes = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic",
           "im_Parabasal", "im_Superficial_Intermediate"]

root_path = "../dataset/SIPaKMeD/test/"

for class_name in classes:
    data_path = os.path.join(root_path, class_name)

    img_path_list = [
        os.path.join(data_path, filename)
        for filename in os.listdir(data_path)
        if filename.lower().endswith(".bmp")
    ]

    for img_path in img_path_list:
        if not os.path.exists(img_path):
            print(f"Error: File not found - {img_path}")
            continue

        # 使用 with 确保文件正确关闭
        with Image.open(img_path) as img:
            dir_path, img_name = os.path.split(img_path)

            # 新文件名格式：类别_原文件名
            new_name = f"{class_name}_{img_name}"
            new_path = os.path.join(dir_path, new_name)

            if os.path.exists(new_path):
                print(f"Warning: File already exists, skipping - {new_path}")
                continue

            try:
                # 保存重命名后的文件
                img.save(new_path)
                print(f"Renamed: {img_name} -> {new_name}")
                # 删除原文件（仅在保存成功后执行）
                os.remove(img_path)
                print(f"Deleted original file: {img_name}")

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
