import os
from torchvision import transforms

class Config:
    BATCH_SIZE = 64
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.0003
    NUM_CLASSES = 5
    AUX_LOSS_WEIGHT = 0.3
    IMAGE_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../../"))
    IMAGE_PATH = os.path.join(IMAGE_ROOT, "dataset/SIPaKMeD/")
    MODEL_SAVE_PATH = "googlenet.pth"
    TRANSFORM = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

opt = Config()