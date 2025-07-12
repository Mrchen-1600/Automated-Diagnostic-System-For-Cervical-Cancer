# -*- coding= utf-8 -*-
# @Author : 尘小风
# @File : Predict.py
# @software : PyCharm

import os
import torch
from PIL import Image
from torchvision import transforms
from network.GoogleNet.model.Model import googlenet

def setup_device():
    """设置计算设备"""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_image_transform():
    """定义图像预处理转换"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def get_image_paths(data_root):
    """获取测试集图片路径列表"""
    data_path = os.path.join(data_root, "dataset/SIPaKMeD/test/")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集路径 {data_path} 不存在")
    return [os.path.join(data_path, i) for i in os.listdir(data_path) if i.endswith(".bmp")]


def load_model(device, num_classes=5, model_path="../train/googlenet.pth"):
    """加载模型及权重"""
    model = googlenet(num_classes=num_classes, aux_logits=False).to(device)
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
    device = setup_device()
    transform = get_image_transform()
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
    img_path_list = get_image_paths(data_root)
    model = load_model(device)

    classes = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic", "im_Parabasal", "im_Superficial_Intermediate"]
    batch_size = 32
    true_num = 0
    total_num = 0

    for i in range(0, len(img_path_list), batch_size):
        batch_paths = img_path_list[i:i + batch_size]
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




