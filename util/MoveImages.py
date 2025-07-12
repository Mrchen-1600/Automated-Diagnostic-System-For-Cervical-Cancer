# -*- coding= utf-8 -*-

# @Author : 尘小风
# @File : MoveImages.py
# @software : PyCharm

import os
import shutil

def move_images_and_clean_dirs(test_dir):
    # 确保 test 目录存在
    if not os.path.exists(test_dir):
        print(f"目录 {test_dir} 不存在")
        return

    # 遍历 test 目录下的所有子目录
    for root, dirs, files in os.walk(test_dir, topdown=False):
        for file in files:
            # 检查文件是否为bmp图片，可根据需求添加更多图片格式
            if file.lower().endswith(('.bmp', 'jpg', 'jpeg', 'png')):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(test_dir, file)

                base_name, ext = os.path.splitext(file)
                while os.path.exists(dst_path):
                    new_name = f"{base_name}_{ext}"
                    dst_path = os.path.join(test_dir, new_name)
                # 移动图片
                shutil.move(src_path, dst_path)
                print(f"已移动 {src_path} 到 {dst_path}")

        # 删除空的子目录
        if root != test_dir and not os.listdir(root):
            os.rmdir(root)
            print(f"已删除空目录 {root}")

# test 目录路径
test_dir = '../dataset/SIPaKMeD/test'
move_images_and_clean_dirs(test_dir)