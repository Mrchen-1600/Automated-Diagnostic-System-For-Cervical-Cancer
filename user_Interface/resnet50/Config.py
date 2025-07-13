import os
import torch
from torchvision import transforms

class Config:
    # 设备相关
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 数据预处理
    TRANSFORM = {
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }
    DATA_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../"))
    TEST_IMAGE_PATH = os.path.join(DATA_ROOT, "images/test_imgs")
    NUM_CLASSES = 5
    BEST_MODEL_PATH = "ResNet50.pth"
    BATCH_SIZE = 64

opt = Config()