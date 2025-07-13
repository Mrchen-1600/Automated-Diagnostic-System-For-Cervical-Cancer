# -*- coding= utf-8 -*-
# @Author : 尘小风
# @File : VGGnet_train.py
# @software : PyCharm

import os
import json
import sys
import math
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from network.ConvNeXt.model.Model import convnext_tiny
from network.ConvNeXt.config.Config import opt
# 使用混合精度训练，结合单精度（FP32）和半精度（FP16）浮点数，能减少显存占用，加快训练速度
from torch.cuda.amp import GradScaler, autocast


class DataManager:

    """
    DataManager类负责数据集的加载、数据加载器的创建以及类别索引文件的保存。
    """

    def __init__(self):
        assert os.path.exists(opt.IMAGE_PATH), f"{opt.IMAGE_PATH} path does not exist."

    def load_datasets(self):
        train_dataset = datasets.ImageFolder(
            root=os.path.join(opt.IMAGE_PATH, "train"),
            transform=opt.TRANSFORM["train"]
        )
        val_dataset = datasets.ImageFolder(
            root=os.path.join(opt.IMAGE_PATH, "val"),
            transform=opt.TRANSFORM["val"]
        )
        return train_dataset, val_dataset

    # 使用多线程加载数据，并固定内存加速传输
    def create_dataloaders(self, train_dataset, val_dataset):
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=opt.BATCH_SIZE,
            shuffle=True,
            num_workers=opt.NUM_WORKERS,
            pin_memory=opt.PIN_MEMORY,
            persistent_workers=opt.PERSISTENT_WORKERS
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=opt.BATCH_SIZE,
            shuffle=False,
            num_workers=opt.NUM_WORKERS,
            pin_memory=opt.PIN_MEMORY,
            persistent_workers=opt.PERSISTENT_WORKERS
        )
        return train_loader, val_loader

    def save_class_indices(self, train_dataset):
        cell_list = train_dataset.class_to_idx
        cla_dict = {index: key for key, index in cell_list.items()}
        with open(opt.JSON_PATH, 'w') as json_file:
            json.dump(cla_dict, json_file, indent=4)



class ModelManager:

    """
    ModelManager类负责模型的初始化、优化器的选择和学习率调度器的初始化
    """

    def __init__(self, num_step: int, epochs: int):
        self.device = torch.device(opt.DEVICE if torch.cuda.is_available() else "cpu")
        self.model = self._initialize_model(self.device)
        self.params_group = self.get_params_groups(self.model)
        self.optimizer = self._initialize_optimizer(self.params_group)
        self.scheduler = self._initialize_scheduler(self.optimizer, num_step, epochs)
        self.loss_function = torch.nn.CrossEntropyLoss()

    # 迁移学习
    def _initialize_model(self, device):
        model = convnext_tiny(num_classes=opt.NUM_CLASSES).to(device)

        if opt.MODEL_WEIGHT_PATH != "":
            weights_dict = torch.load(opt.MODEL_WEIGHT_PATH, map_location=opt.DEVICE)["model"]
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

        if opt.FREEZE_LAYERS:
            for name, para in model.named_parameters():
                # 除head外，其他权重全部冻结
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))
        return model


    # 将模型的可训练参数划分为应用权重衰减和不应用权重衰减的两组，帮助优化器对不同类型的参数采用不同的训练策略
    def get_params_groups(self, model):
        # 记录optimize要训练的权重参数
        parameter_group_vars = {"decay": {"params": [], "weight_decay": opt.WEIGHT_DECAY},
                                "no_decay": {"params": [], "weight_decay": 0.}}

        # 记录对应的权重名称
        parameter_group_names = {"decay": {"params": [], "weight_decay": opt.WEIGHT_DECAY},
                                 "no_decay": {"params": [], "weight_decay": 0.}}

        for name, param in model.named_parameters():
            # 如果参数不需要梯度更新（即被冻结），则跳过该参数
            if not param.requires_grad:
                continue

                # 如果参数的维度为 1 或者参数名称以 .bias 结尾，将其划分到 no_decay 组，否则划分到 decay 组
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
            else:
                group_name = "decay"

            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)

        print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
        return list(parameter_group_vars.values())


    # 根据配置文件的配置，初始化相应的优化器
    def _initialize_optimizer(self, params_group):
        # params = [p for p in self.model.parameters() if p.requires_grad]

        if opt.OPTIMIZER_TYPE == "SGD":
            optimizer = torch.optim.SGD(
                params_group,
                lr=opt.LEARNING_RATE,
                momentum=opt.MOMENTUM,
                weight_decay=opt.WEIGHT_DECAY
            )
        elif opt.OPTIMIZER_TYPE == "Adam":
            optimizer = torch.optim.Adam(
                params_group,
                lr=opt.LEARNING_RATE,
                weight_decay = opt.WEIGHT_DECAY
            )
        elif opt.OPTIMIZER_TYPE == "AdamW":
            optimizer = torch.optim.AdamW(
                params_group,
                lr=opt.LEARNING_RATE,
                weight_decay=opt.WEIGHT_DECAY
            )
        else:
            raise ValueError("Unsupported optimizer type. Choose 'SGD' or 'Adam'.")
        return optimizer

    # 余弦退火学习率调度器，用于动态调整学习率。
    # 该调度器根据当前训练步骤数和总训练步骤数，计算出一个学习率倍率因子，然后根据该因子调整学习率。
    def _initialize_scheduler(self, optimizer,
                            num_step: int,
                            epochs: int,
                            warmup=opt.WARMUP,
                            warmup_epochs=opt.WARMUP_EPOCHS,
                            warmup_factor=opt.WARMUP_FACTOR,
                            end_factor=opt.END_FACTOR):
        assert num_step > 0 and epochs > 0
        if warmup is False:
            warmup_epochs = 0

        def f(x):
            """
            根据step数返回一个学习率倍率因子，
            注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
            """
            if warmup is True and x <= (warmup_epochs * num_step):
                alpha = float(x) / (warmup_epochs * num_step)
                # warmup过程中lr倍率因子从warmup_factor -> 1
                return warmup_factor * (1 - alpha) + alpha
            else:
                current_step = (x - warmup_epochs * num_step)
                cosine_steps = (epochs - warmup_epochs) * num_step
                # warmup后lr倍率因子从1 -> end_factor
                return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

        return lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


class Trainer:

    """
    Trainer类负责训练过程的管理，包括单个 epoch 的训练、验证以及训练日志的记录和模型保存
    """

    def __init__(self, model_manager, train_loader, val_loader):
        self.model_manager = model_manager
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tb_writer = SummaryWriter(opt.RUNS_DIR)
        if not os.path.exists(opt.WEIGHTS_DIR):
            os.makedirs(opt.WEIGHTS_DIR)
        self.scaler = GradScaler()  # 初始化梯度缩放器

    def train_one_epoch(self, epoch):
        self.model_manager.model.train()
        accu_loss = torch.zeros(1).to(self.model_manager.device) # 累计损失
        accu_num = torch.zeros(1).to(self.model_manager.device) # 累计预测正确的样本数
        self.model_manager.optimizer.zero_grad()

        sample_num = 0
        # 训练进度条
        data_loader = tqdm(self.train_loader, file=sys.stdout)
        for step, data in enumerate(data_loader):
            images, labels = data
            sample_num += images.shape[0]

            with autocast():  # 开启自动混合精度
                pred = self.model_manager.model(images.to(self.model_manager.device))
                pred_classes = torch.max(pred, dim=1)[1]
                accu_num += torch.eq(pred_classes, labels.to(self.model_manager.device)).sum()

                loss = self.model_manager.loss_function(pred, labels.to(self.model_manager.device))

            self.scaler.scale(loss).backward()  # 缩放损失并反向传播
            accu_loss += loss.detach()

            data_loader.desc = f"[train epoch {epoch}] loss: {accu_loss.item() / (step + 1):.3f}, acc: {accu_num.item() / sample_num:.3f}"

            # 检查损失值是否为有限值，如果不是，则打印警告信息并退出程序
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            self.scaler.step(self.model_manager.optimizer)  # 缩放优化器步骤
            self.scaler.update()  # 更新缩放器
            # 优化器梯度清零
            self.model_manager.optimizer.zero_grad()

            # 释放不再需要的变量
            del images, labels, pred, loss
            torch.cuda.empty_cache()  # 释放GPU缓存

        return accu_loss.item() / (step + 1), accu_num.item() / sample_num


    def evaluate(self, epoch):
        self.model_manager.model.eval()
        accu_num = torch.zeros(1).to(self.model_manager.device)
        accu_loss = torch.zeros(1).to(self.model_manager.device)

        sample_num = 0

        with torch.no_grad():
            data_loader = tqdm(self.val_loader, file=sys.stdout)
            for step, data in enumerate(data_loader):
                images, labels = data
                sample_num += images.shape[0]

                with autocast():  # 开启自动混合精度
                    pred = self.model_manager.model(images.to(self.model_manager.device))
                    pred_classes = torch.max(pred, dim=1)[1]
                    accu_num += torch.eq(pred_classes, labels.to(self.model_manager.device)).sum()

                    loss = self.model_manager.loss_function(pred, labels.to(self.model_manager.device))

                accu_loss += loss

                data_loader.desc = f"[val epoch {epoch}] loss: {accu_loss.item() / (step + 1):.3f}, acc: {accu_num.item() / sample_num:.3f}"
                # 释放不再需要的变量
                del images, labels, pred, loss
                torch.cuda.empty_cache()  # 释放GPU缓存

            return accu_loss.item() / (step + 1), accu_num.item() / sample_num


    def run_training(self):
        best_val_acc = 0.0

        for epoch in range(opt.EPOCHS):
            # 训练和验证
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.evaluate(epoch)

            # 在 Trainer 类的 run_training 方法中修改如下：
            self.tb_writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch)
            self.tb_writer.add_scalars('Accuracy', {'Train': train_acc, 'Validation': val_acc}, epoch)
            self.tb_writer.add_scalar('Learning Rate',self.model_manager.optimizer.param_groups[0]["lr"], epoch)


            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model_manager.model.state_dict(),
                           opt.BEST_MODEL_PATH)


            if self.model_manager.scheduler:
                self.model_manager.scheduler.step()

        # 训练完成后关闭TensorBoard写入器
        self.tb_writer.close()


def main():
    # 打印关键配置信息用于调试
    print(f"Using device: {opt.DEVICE}")
    print(f"Train transform: {opt.TRANSFORM['train']}")
    print(f"Val transform: {opt.TRANSFORM['val']}")

    # 初始化数据管理
    data_manager = DataManager()
    train_dataset, val_dataset = data_manager.load_datasets()
    train_loader, val_loader = data_manager.create_dataloaders(train_dataset, val_dataset)
    data_manager.save_class_indices(train_dataset)

    # 打印数据集信息
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")


    model_manager = ModelManager(len(train_loader), opt.EPOCHS)
    trainer = Trainer(model_manager, train_loader, val_loader)
    trainer.run_training()


if __name__ == "__main__":
    main()