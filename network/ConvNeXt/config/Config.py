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
    # 迁移学习预训练权重文件，如果不想载入预训练权重，就设置成""
    MODEL_WEIGHT_PATH = "./convnext_tiny_1k_224_ema.pth"
    BEST_MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_model.pth")
    JSON_PATH = "class_indices.json"
    CLASSES = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic", "im_Parabasal", "im_Superficial_Intermediate"]

    # 训练参数
    BATCH_SIZE = 64
    EPOCHS = 20
    NUM_CLASSES = 5
    # 是否冻结head以外所有权重
    # 当使用预训练权重时，可以把这个值设置成true，表示只对分类层做微调
    FREEZE_LAYERS = False

    # 优化器相关
    OPTIMIZER_TYPE = "AdamW"  # 可选择 "SGD" 或 "Adam" 或 "AdamW"
    LEARNING_RATE = 5e-2
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4

    # 学习率调度器相关
    USE_SCHEDULER = False
    GAMMA = 0.9
    # 余弦退火学习率调度器相关
    WARMUP = True # 是否开启热身训练机制。热身训练是在训练初期逐渐增加学习率的一种策略，有助于模型在训练开始时更稳定地收敛
    WARMUP_EPOCHS = 2 # 热身训练的周期数。当 warmup为True时，模型会在这几个周期内逐渐将学习率从warmup_factor提升到初始学习率
    WARMUP_FACTOR = 1e-3 # 热身训练开始时学习率的倍率因子。初始学习率会乘以这个因子作为热身阶段的起始学习率
    END_FACTOR = 1e-6 # 训练结束时学习率的倍率因子。在热身阶段结束后，学习率会通过余弦退火策略逐渐衰减到初始学习率乘以end_factor的值

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