from torchvision import transforms

class Config():
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 0.002
    NUM_CLASSES = 5
    IMAGE_PATH =  "../../dataset/SIPaKMeD/"
    JSON_PATH = "../class_indices.json"
    MODEL_SAVE_PATH = "alexnet.pth"
    TRANSFORM = {
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

opt = Config()