import os
import torch
from torchvision import transforms

class Config:
    # 设备相关
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = min(4, os.cpu_count()) # 使用最多4个工作进程
    PIN_MEMORY = True if DEVICE == "cuda:0" else False # 固定内存加速GPU传输
    PERSISTENT_WORKERS = True if NUM_WORKERS > 0 else False # 保持工作进程活跃

    # 路径相关
    WEIGHTS_DIR = "../output/weights"
    RUNS_DIR = "../output/runs"
    DATA_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../../"))
    IMAGE_PATH = os.path.join(DATA_ROOT, "dataset/SIPaKMeD/")
    TEST_IMAGE_PATH = os.path.join(IMAGE_PATH, "test")
    MODEL_WEIGHT_PATH = "./resnet50-pre.pth" # 迁移学习预训练权重文件
    BEST_MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_model.pth")
    JSON_PATH = "class_indices.json"
    CLASSES = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic", "im_Parabasal", "im_Superficial_Intermediate"]

    # 训练参数
    BATCH_SIZE = 16
    EPOCHS = 3
    NUM_CLASSES = 5

    # 优化器相关
    OPTIMIZER_TYPE = "Adam"  # 可选择 "SGD" 或 "Adam"
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-5

    # 学习率调度器相关
    USE_SCHEDULER = False
    GAMMA = 0.9

    # 数据预处理
    TRANSFORM = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),

        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),

        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }

opt = Config()