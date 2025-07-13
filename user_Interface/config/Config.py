# 配置文件 config.py
import torch
from torchvision import transforms


class Config:
    # 设备配置
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 图像预处理配置
    TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 模型路径配置
    MODEL_PATHS = {
        "ViT": "visiontransformer/VisionTransformer.pth",
        "VGG-16": "vgg16/VGG16.pth",
        "ResNet-50": "resnet50/ResNet50.pth"
    }

    # 类别配置
    CLASSES_NAMES = ['角化不良细胞', '挖空细胞', '化生细胞', '副基底层细胞', '浅表中层细胞']
    CLASSES_ENGLISH = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic", "im_Parabasal",
                       "im_Superficial_Intermediate"]

    # 界面配置
    WINDOW_TITLE = '基于深度学习的宫颈细胞在线分类识别系统V1.0'
    WINDOW_SIZE = (1200, 800)
    ICON_PATH = "images/UI/school.jpg"
    OUTPUT_IMG_SIZE = 448
    FONT_TITLE = ('楷体', 22)
    FONT_MAIN = ('楷体', 16)

    # 批量处理配置
    BATCH_SIZE = 8

    # 置信度阈值
    CONFIDENCE_THRESHOLD = 0.9

    # 样式配置
    BUTTON_STYLE = """
        QPushButton {
            color: white;
            background-color: rgb(5,100,195);
            border-width: 4px;
            border-color: rgb(5,100,195);
            border-radius: 4px;
            padding: 8px 8px;
            margin: 4px 4px;
            font: 16pt '楷体';
        }
        QPushButton:hover {
            background-color: rgb(5,100,195);
        }
        QPushButton:pressed {
            background-color: rgb(85,170,255);
        }
    """

    # 链接样式
    LINK_STYLE = """
        QLabel {
            color: blue;
            text-decoration: underline;
            font: 14pt '楷体';
        }
    """

cfg = Config()