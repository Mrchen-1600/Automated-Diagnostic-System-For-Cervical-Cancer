# -*- coding= utf-8 -*-
# @Author : 尘小风
# @File : VGGNet_predict（批量预测）.py
# @software : PyCharm

import os
import torch
from PIL import Image
from network.VGGNet.model.Model import vgg
from network.VGGNet.config.Config import  opt


def get_image_paths(test_image_path):
    """获取测试集图片路径列表"""
    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"数据集路径 {test_image_path} 不存在")
    return [os.path.join(test_image_path, i) for i in os.listdir(test_image_path) if i.endswith(".bmp")]


def load_model(model_name, device, num_classes, model_path):
    """加载模型及权重"""
    model = vgg(model_name=model_name, num_classes=num_classes, batch_norm=True, init_weights=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_batch(images, model, device):
    """批量预测"""
    images = torch.stack(images).to(device)
    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu()
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    return probs, predictions


def main():
    device = opt.DEVICE
    transform = opt.TRANSFORM["test"]
    test_image_path = opt.TEST_IMAGE_PATH
    img_path_list = get_image_paths(test_image_path)
    model = load_model(opt.MODEL_NAME, device, opt.NUM_CLASSES, opt.BEST_MODEL_PATH)

    classes = opt.CLASSES

    true_num = 0
    total_num = 0
    for i in range(0, len(img_path_list), opt.BATCH_SIZE):
        batch_paths = img_path_list[i:i + opt.BATCH_SIZE]
        batch_images = []
        batch_labels = []

        for img_path in batch_paths:
            try:
                img = Image.open(img_path)
                path, img_name = os.path.split(img_path)
                img = transform(img)
                batch_images.append(img)
                # 之前已经对测试集图像重命名过，因此我们可以直接获取图像的名称从而知道图像的真实类别
                batch_labels.append(img_name[:-11])
            except Exception as e:
                print(f"处理图片 {img_path} 时出错: {e}")

        if batch_images:
            probs, predictions = predict_batch(batch_images, model, device)
            total_num += len(batch_images)
            for j in range(len(batch_images)):
                if batch_labels[j] == classes[predictions[j]]:
                    true_num += 1

    if total_num == 0:
        print("没有有效的图片进行预测，请检查数据集路径和图片格式。")
    else:
        accuracy = 100 * true_num / total_num
        print(f"The accuracy of prediction is {accuracy:.3f}%")

if __name__ == "__main__":
    main()
