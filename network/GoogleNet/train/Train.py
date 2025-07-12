# -*- coding= utf-8 -*-
# @Author : 尘小风
# @File : Train.py
# @software : PyCharm

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
from network.GoogleNet.model.Model import Model
from tqdm import tqdm
from network.GoogleNet.config.Config import opt


# 加载数据集
def load_datasets(image_path):
    transform = opt.TRANSFORM
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=transform["train"])
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=transform["val"])
    return train_dataset, val_dataset


# 加载数据加载器
def load_dataloaders(train_dataset, val_dataset, batch_size):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)  # 验证集无需打乱
    print(f"using {len(train_dataset)} images for training, {len(val_dataset)} images for val")
    return train_loader, val_loader


def train_one_epoch(model, train_loader, loss_function, optimizer, loss_list):
    """
    训练一个 epoch

    Args:
        model (torch.nn.Module): 模型
        train_loader (DataLoader): 训练集数据加载器
        loss_function (torch.nn.Module): 损失函数
        optimizer (torch.optim.Optimizer): 优化器
        loss_list (list): 用于记录每个批次的损失值

    Returns:
        float: 该 epoch 的平均损失
    """
    model.train()
    loss_sum = 0
    for i, (inputs, targets) in enumerate(tqdm(train_loader, desc="Training")):
        logits, logits_aux2, logits_aux1 = model(inputs)
        loss0 = loss_function(logits, targets)
        loss1 = loss_function(logits_aux1, targets)
        loss2 = loss_function(logits_aux2, targets)
        loss = loss0 + opt.AUX_LOSS_WEIGHT * loss1 + opt.AUX_LOSS_WEIGHT * loss2

        loss_sum += loss.item()
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_sum / len(train_loader)


def validate_one_epoch(model, val_loader):
    """
    验证一个epoch

    Args:
        model (torch.nn.Module): 模型
        val_loader (DataLoader): 验证集数据加载器

    Returns:
        float: 验证集准确率
    """
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            y_pred = model(images)
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]  # 取主输出
            _, predicted = torch.max(y_pred.data, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def plot_results(loss_list, accuracy_list):
    """
    绘制训练损失和验证准确率曲线

    Args:
        loss_list (list): 训练损失列表
        accuracy_list (list): 验证准确率列表
    """
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

    model = Model(
        num_classes=opt.NUM_CLASSES,
        aux_logits=True,
        init_weights=True
    )
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.LEARNING_RATE)

    loss_list = []
    accuracy_list = []
    best_acc = 0

    for epoch in range(opt.NUM_EPOCHS):
        avg_loss = train_one_epoch(model, train_loader, loss_function, optimizer, loss_list)
        val_accuracy = validate_one_epoch(model, val_loader)

        accuracy_list.append(val_accuracy)
        print(f"[epoch {epoch + 1}], train_loss:{avg_loss:.3f}, val_accuracy:{val_accuracy * 100:.3f}% ")

        if val_accuracy > best_acc:
            best_acc =val_accuracy
            torch.save(model.state_dict(), opt.MODEL_SAVE_PATH)

    print("Finishing Training")
    plot_results(loss_list, accuracy_list)

if __name__ == "__main__":
    main()