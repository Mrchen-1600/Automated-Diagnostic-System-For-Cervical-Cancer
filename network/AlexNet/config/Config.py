class Config():
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 0.002
    NUM_CLASSES = 5
    IMAGE_PATH =  "../../dataset/SIPaKMeD/"
    JSON_PATH = "../class_indices.json"
    MODEL_SAVE_PATH = "alexnet.pth"

opt = Config()