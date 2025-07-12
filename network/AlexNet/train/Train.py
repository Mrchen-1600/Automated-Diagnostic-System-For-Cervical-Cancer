# -*- coding= utf-8 -*-
# @Time : 2023/3/26 16:16
# @Author : 尘小风
# @File : Train.py
# @software : PyCharm

import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from network.AlexNet.config.Config import opt
from network.AlexNet.model.Model import Model
import json
import os
import time
from tqdm import tqdm



# 数据预处理
def get_transforms():
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

# 加载数据集
def load_datasets(image_path):
    transform = get_transforms()
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=transform["train"])
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=transform["val"])
    return train_dataset, val_dataset


# 加载数据加载器
def load_dataloaders(train_dataset, val_dataset, batch_size):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)  # 验证集无需打乱
    return train_loader, val_loader


# 保存类别索引
def save_class_indices(train_dataset, json_path):
    cell_list = train_dataset.class_to_idx
    cla_dict = {index: key for key, index in cell_list.items()}
    with open(json_path, 'w') as json_file:
        json.dump(cla_dict, json_file, indent=4)


# 训练一个 epoch
def train_one_epoch(model, train_loader, criterion, optimizer, loss_list):
    model.train()
    loss_sum = 0
    for inputs, targets in tqdm(train_loader, desc="Training"):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss_list.append(loss.item())
        loss_sum += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_sum

# 验证一个 epoch
def validate_one_epoch(model, val_loader, best_acc, model_save_path):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            y_pred = model(images)
            _, predicted = torch.max(y_pred, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), model_save_path)
    return best_acc, accuracy

# 绘制训练曲线
def plot_training_curves(loss_list, accuracy_list):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.subplot(122)
    plt.plot(range(len(accuracy_list)), accuracy_list)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

def main():
    # 加载数据集和数据加载器
    train_dataset, val_dataset = load_datasets(opt.IMAGE_PATH)
    train_loader, val_loader = load_dataloaders(train_dataset, val_dataset, opt.BATCH_SIZE)

    # 保存类别索引
    save_class_indices(train_dataset, opt.JSON_PATH)

    # 初始化模型、损失函数和优化器
    model = Model(num_classes=opt.NUM_CLASSES)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.LEARNING_RATE)

    loss_list = []
    accuracy_list = []
    best_acc = 0

    for epoch in range(opt.EPOCHS):
        time_start = time.perf_counter()

        # 训练
        loss_sum = train_one_epoch(model, train_loader, criterion, optimizer, loss_list)

        # 验证
        best_acc, accuracy = validate_one_epoch(model, val_loader, best_acc, opt.MODEL_SAVE_PATH)
        accuracy_list.append(accuracy)

        print(f"\n[Epoch {epoch + 1}] Train Loss: {loss_sum / len(train_loader):.3f} Val Accuracy: {accuracy:.3f}")
        print(f"Time taken: {time.perf_counter() - time_start:.2f} s")

    print("Finished Training")

    # 绘制训练曲线
    plot_training_curves(loss_list, accuracy_list)

if __name__ == "__main__":
    main()

