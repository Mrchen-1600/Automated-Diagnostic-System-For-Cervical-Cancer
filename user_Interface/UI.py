# -*- coding= utf-8 -*-
# @Author : 尘小风
# @File : Model.py
# @software : PyCharm


from config.Config import cfg
import os
import sys
import os.path as osp
import shutil
import cv2
import torch
from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import datetime

# 导入模型
from visiontransformer.VisionTransformer_Model import vit_base_patch16_224_in21k as create_model
from vgg16.VGGNet_Model import vgg
from resnet50.ResNet50_Model import resnet50


class ModelPredictor:
    def __init__(self):
        self.models = {}
        self.report_path = ""

    def load_model(self, model_name):
        """加载模型并缓存"""
        if model_name in self.models:
            return self.models[model_name]

        model_path = cfg.MODEL_PATHS[model_name]

        if model_name == "ViT":
            model = create_model(num_classes=5, has_logits=False).to(cfg.DEVICE)
        elif model_name == "VGG-16":
            model = vgg(model_name="vgg16", num_classes=5, batch_norm=True, init_weights=False).to(cfg.DEVICE)
        elif model_name == "ResNet-50":
            model = resnet50(num_classes=5).to(cfg.DEVICE)
        else:
            raise ValueError(f"未知模型: {model_name}")

        model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
        model.eval()
        self.models[model_name] = model
        return model

    def predict_single(self, model_name, img_path):
        """单张图片预测"""
        model = self.load_model(model_name)
        img = Image.open(img_path)
        img = cfg.TRANSFORM(img)
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            output = model(img.to(cfg.DEVICE))

        # 计算概率
        probabilities = torch.softmax(output, dim=1)[0]
        predict_cla = torch.argmax(output).item()
        confidence = probabilities[predict_cla].item()

        return {
            "class_name": cfg.CLASSES_NAMES[predict_cla],
            "confidence": confidence,
            "file_name": osp.basename(img_path)
        }

    def predict_batch(self, model_name, img_paths):
        """批量图片预测"""
        model = self.load_model(model_name)
        total = len(img_paths)
        results = []
        class_stats = {name: {"total": 0, "uncertain": 0, "files": []} for name in cfg.CLASSES_NAMES}

        # 确保有图片需要处理
        if total == 0:
            return class_stats, results

        # 批量处理图片
        for i in range(0, total, cfg.BATCH_SIZE):
            batch_paths = img_paths[i:i + cfg.BATCH_SIZE]
            batch_images = []
            batch_info = []

            # 准备batch数据
            for img_path in batch_paths:
                img = Image.open(img_path)
                img_tensor = cfg.TRANSFORM(img)
                batch_images.append(img_tensor)

                # 保存图片信息
                batch_info.append({
                    "path": img_path,
                    "file_name": osp.basename(img_path)
                })

            # 转换为tensor
            batch_tensor = torch.stack(batch_images).to(cfg.DEVICE)

            # 批量预测
            with torch.no_grad():
                outputs = model(batch_tensor)

            # 计算概率
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
            confidences = torch.max(probabilities, dim=1).values.cpu().numpy()

            # 处理每个图片的结果
            for j in range(len(batch_paths)):
                pred_idx = predictions[j]
                confidence = confidences[j]
                class_name = cfg.CLASSES_NAMES[pred_idx]

                # 更新统计信息
                class_stats[class_name]["total"] += 1

                # 检查置信度
                is_uncertain = confidence < cfg.CONFIDENCE_THRESHOLD
                if is_uncertain:
                    class_stats[class_name]["uncertain"] += 1
                    class_stats[class_name]["files"].append(batch_info[j]["file_name"])

                # 保存结果
                results.append({
                    "file_name": batch_info[j]["file_name"],
                    "predicted_class": class_name,
                    "confidence": confidence,
                    "is_uncertain": is_uncertain
                })

        # 生成报告
        self.generate_report(model_name, class_stats, results)

        return class_stats, results

    def generate_report(self, model_name, class_stats, results):
        """生成识别报告"""
        # 创建报告目录
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)

        # 生成文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{report_dir}/report_{model_name}_{timestamp}.txt"

        # 写入报告内容
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"宫颈细胞识别报告 - {model_name}模型\n")
            f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")

            # 总体统计
            total_images = sum(stats["total"] for stats in class_stats.values())
            uncertain_images = sum(stats["uncertain"] for stats in class_stats.values())
            f.write(f"总图片数量: {total_images}\n")
            f.write(f"不确定图片数量: {uncertain_images} (置信度 < {cfg.CONFIDENCE_THRESHOLD * 100}%)\n")
            f.write("\n")

            # 各类别统计
            f.write("各类别统计:\n")
            f.write("-" * 50 + "\n")
            for class_name, stats in class_stats.items():
                f.write(f"类别: {class_name}\n")
                f.write(f"  总数量: {stats['total']}\n")
                f.write(f"  不确定数量: {stats['uncertain']}\n")

                if stats["uncertain"] > 0:
                    f.write("  不确定图片列表:\n")
                    for i, file_name in enumerate(stats["files"], 1):
                        f.write(f"    {i}. {file_name}\n")
                f.write("\n")

            # 所有图片结果
            f.write("\n详细识别结果:\n")
            f.write("-" * 50 + "\n")
            for result in results:
                status = "不确定" if result["is_uncertain"] else "确定"
                f.write(f"图片: {result['file_name']}\n")
                f.write(f"  预测类别: {result['predicted_class']}\n")
                f.write(f"  置信度: {result['confidence']:.4f} ({status})\n")

        self.report_path = report_file
        return report_file


# UI主界面
class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        # 使用配置初始化界面
        self.setWindowTitle(cfg.WINDOW_TITLE)
        self.resize(*cfg.WINDOW_SIZE)
        self.setWindowIcon(QIcon(cfg.ICON_PATH))

        # 图片相关属性
        self.output_size = cfg.OUTPUT_IMG_SIZE
        self.predictimg = ""
        self.predictbatch = []
        self.origin_shape = ()

        # 初始化预测器
        self.predictor = ModelPredictor()

        # 初始化界面
        self.initUI()
        self.batch_initUI()
        self.developer_initUI()

    def initUI(self):
        # 图片识别子界面
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()

        # 标题
        title = QLabel("宫颈细胞图片分类识别系统")
        title.setFont(QFont(*cfg.FONT_TITLE))

        # 封面图片
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.fengmian_img = QLabel()
        self.fengmian_img.setPixmap(QPixmap("images/UI/cover.png"))
        self.fengmian_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.fengmian_img)
        mid_img_widget.setLayout(mid_img_layout)

        # 按钮
        up_img_button = QPushButton("上传待检测图片")
        det_img_button = QPushButton("一键识别")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)

        # 提示文本
        self.Prompt_text = QLabel("等待上传")
        self.Prompt_text.setFont(QFont(*cfg.FONT_MAIN))

        # 报告下载链接（初始隐藏）
        self.report_link = QLabel()
        self.report_link.setFont(QFont(*cfg.FONT_MAIN))
        self.report_link.setStyleSheet(cfg.LINK_STYLE)
        self.report_link.setVisible(False)
        self.report_link.mousePressEvent = self.download_report

        # 应用样式
        for btn in [up_img_button, det_img_button]:
            btn.setStyleSheet(cfg.BUTTON_STYLE)

        # 布局
        img_detection_layout.addWidget(title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(self.Prompt_text)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_layout.addWidget(self.report_link)
        img_detection_widget.setLayout(img_detection_layout)

        self.addTab(img_detection_widget, '图片识别')
        self.setTabIcon(0, QIcon(cfg.ICON_PATH))

    def batch_initUI(self):
        # 批量识别子界面
        batch_detection_widget = QWidget()
        batch_detection_layout = QVBoxLayout()

        # 标题
        title = QLabel("宫颈细胞批量分类识别系统")
        title.setFont(QFont(*cfg.FONT_TITLE))

        # 封面图片
        mid_img_widget2 = QWidget()
        mid_img_layout2 = QHBoxLayout()
        self.fengmian_img2 = QLabel()
        self.fengmian_img2.setPixmap(QPixmap("images/UI/cover.png"))
        self.fengmian_img2.setAlignment(Qt.AlignCenter)
        mid_img_layout2.addWidget(self.fengmian_img2)
        mid_img_widget2.setLayout(mid_img_layout2)

        # 按钮
        up_img_button2 = QPushButton("选择存放待检测图片的文件夹")
        det_img_button1 = QPushButton("VGG识别")
        det_img_button2 = QPushButton("ResNet识别")
        det_img_button3 = QPushButton("ViT识别")

        up_img_button2.clicked.connect(self.upload_batch)
        det_img_button1.clicked.connect(self.detect_VGG)
        det_img_button2.clicked.connect(self.detect_ResNet)
        det_img_button3.clicked.connect(self.detect_ViT)

        # 提示文本
        self.Prompt_label = QLabel("等待上传")
        self.Prompt_label.setFont(QFont(*cfg.FONT_MAIN))

        # 统计信息显示
        self.stats_label = QLabel()
        self.stats_label.setFont(QFont(*cfg.FONT_MAIN))
        self.stats_label.setVisible(False)

        # 报告下载链接
        self.batch_report_link = QLabel()
        self.batch_report_link.setFont(QFont(*cfg.FONT_MAIN))
        self.batch_report_link.setStyleSheet(cfg.LINK_STYLE)
        self.batch_report_link.setVisible(False)
        self.batch_report_link.mousePressEvent = self.download_report

        # 应用样式
        buttons = [up_img_button2, det_img_button1, det_img_button2, det_img_button3]
        for btn in buttons:
            btn.setStyleSheet(cfg.BUTTON_STYLE)

        # 布局
        batch_detection_layout.addWidget(title, alignment=Qt.AlignCenter)
        batch_detection_layout.addWidget(mid_img_widget2, alignment=Qt.AlignCenter)
        batch_detection_layout.addWidget(self.Prompt_label)
        batch_detection_layout.addWidget(up_img_button2)
        batch_detection_layout.addWidget(det_img_button1)
        batch_detection_layout.addWidget(det_img_button2)
        batch_detection_layout.addWidget(det_img_button3)
        batch_detection_layout.addWidget(self.stats_label)
        batch_detection_layout.addWidget(self.batch_report_link)
        batch_detection_widget.setLayout(batch_detection_layout)

        self.addTab(batch_detection_widget, '批量识别')
        self.setTabIcon(1, QIcon(cfg.ICON_PATH))

    def developer_initUI(self):
        # 开发信息子界面
        develop_widget = QWidget()
        develop_layout = QVBoxLayout()

        # 标题
        title = QLabel('欢迎使用基于深度学习的宫颈细胞在线分类系统')
        title.setFont(QFont('楷体', 24))
        title.setAlignment(Qt.AlignCenter)

        # 图片链接
        college_sign = QLabel()
        college_sign.setText(
            '<a href="https://github.com/Mrchen-1600/Automated-Diagnostic-System-For-Cervical-Cancer">')
        college_sign.setPixmap(QPixmap(cfg.ICON_PATH))
        college_sign.setAlignment(Qt.AlignCenter)
        college_sign.setOpenExternalLinks(True)

        # 开发者信息
        developer = QLabel(
            '<a href="https://github.com/Mrchen-1600/Automated-Diagnostic-System-For-Cervical-Cancer">开发者：浙江大学 MrChen')
        developer.setFont(QFont('楷体', 16))
        developer.setOpenExternalLinks(True)

        # 布局
        develop_layout.addWidget(title)
        develop_layout.addStretch(2)
        develop_layout.addWidget(college_sign)
        develop_layout.addStretch()
        develop_layout.addWidget(developer)
        develop_widget.setLayout(develop_layout)

        self.addTab(develop_widget, '开发信息')
        self.setTabIcon(2, QIcon(cfg.ICON_PATH))

    # 上传待检测图片
    def upload_img(self):
        fileName, _ = QFileDialog.getOpenFileName(self, '选择图片', '', '*.jpg *.png *.bmp')
        if not fileName:
            return

        suffix = fileName.split(".")[-1]
        save_path = osp.join("images/tmp", f"upload.{suffix}")
        shutil.copy(fileName, save_path)

        # 调整图片大小
        im0 = cv2.imread(save_path)
        resize_scale = self.output_size / im0.shape[0]
        im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
        cv2.imwrite("images/tmp/upload_enlarger.bmp", im0)

        self.predictimg = fileName
        self.origin_shape = (im0.shape[1], im0.shape[0])

        # 更新界面
        self.fengmian_img.setPixmap(QPixmap("images/tmp/upload_enlarger.bmp"))
        self.Prompt_text.setText("点击'一键识别'按钮")

        # 隐藏之前的报告链接
        self.report_link.setVisible(False)

    # 单一图片分类识别
    def detect_img(self):
        if not self.predictimg:
            self.Prompt_text.setText("请先上传图片")
            return

        try:
            result = self.predictor.predict_single("ViT", self.predictimg)
            confidence_percent = result["confidence"] * 100

            # 检查置信度
            if confidence_percent < cfg.CONFIDENCE_THRESHOLD * 100:
                status = "不确定，需要人工复检"
            else:
                status = "确定"

            self.Prompt_text.setText(
                f"识别结果: {result['class_name']}\n"
                f"置信度: {confidence_percent:.2f}% ({status})"
            )

            # 生成简单报告
            self.generate_single_report(result)
            self.report_link.setText('<a href="#">下载识别报告</a>')
            self.report_link.setVisible(True)

        except Exception as e:
            self.Prompt_text.setText(f"识别出错: {str(e)}")
            self.report_link.setVisible(False)

    def generate_single_report(self, result):
        """生成单张图片识别报告"""
        # 创建报告目录
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)

        # 生成文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{report_dir}/single_report_{timestamp}.txt"

        # 写入报告内容
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"宫颈细胞识别报告 - 单张图片\n")
            f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"图片名称: {result['file_name']}\n")
            f.write(f"预测类别: {result['class_name']}\n")
            f.write(f"置信度: {result['confidence'] * 100:.2f}%\n")

            # 判断是否需要复检
            if result["confidence"] < cfg.CONFIDENCE_THRESHOLD:
                f.write("\n识别结果置信度较低，建议人工复检\n")
            else:
                f.write("\n识别结果置信度较高，可信任\n")

        self.predictor.report_path = report_file

    # 选择待检测文件夹
    def upload_batch(self):
        data_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹", "./")
        if not data_path:
            return

        self.predictbatch = [
            osp.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".bmp")
        ]

        count = len(self.predictbatch)
        self.Prompt_label.setText(f"已选择 {count} 张图片，请选择识别模型")

        # 隐藏之前的统计和报告
        self.stats_label.setVisible(False)
        self.batch_report_link.setVisible(False)

    # 批量识别方法
    def _batch_detect(self, model_name):
        if not self.predictbatch:
            self.Prompt_label.setText("请先选择图片文件夹")
            return

        try:
            # 执行批量识别
            class_stats, _ = self.predictor.predict_batch(model_name, self.predictbatch)

            # 更新界面显示统计信息
            stats_text = f"{model_name}模型识别结果:\n"
            stats_text += "=" * 30 + "\n"

            for class_name, stats in class_stats.items():
                stats_text += (
                    f"{class_name}: {stats['total']}张, "
                    f"其中{stats['uncertain']}张不确定\n"
                )

            # 添加总结
            total_images = sum(stats["total"] for stats in class_stats.values())
            uncertain_images = sum(stats["uncertain"] for stats in class_stats.values())
            stats_text += "\n"
            stats_text += f"总图片: {total_images}张\n"
            stats_text += f"不确定图片: {uncertain_images}张 (需人工复检)\n"

            self.stats_label.setText(stats_text)
            self.stats_label.setVisible(True)

            # 显示报告下载链接
            self.batch_report_link.setText('<a href="#">下载完整识别报告</a>')
            self.batch_report_link.setVisible(True)

        except Exception as e:
            self.Prompt_label.setText(f"{model_name}识别出错: {str(e)}")
            self.stats_label.setVisible(False)
            self.batch_report_link.setVisible(False)

    def detect_ViT(self):
        self._batch_detect("ViT")

    def detect_VGG(self):
        self._batch_detect("VGG-16")

    def detect_ResNet(self):
        self._batch_detect("ResNet-50")

    def download_report(self, event):
        """下载报告文件"""
        if not self.predictor.report_path or not osp.exists(self.predictor.report_path):
            QMessageBox.warning(self, "错误", "报告文件不存在或尚未生成")
            return

        # 获取用户选择的保存位置
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存报告",
            osp.basename(self.predictor.report_path),
            "文本文件 (*.txt)"
        )

        if not save_path:
            return

        # 复制报告文件
        try:
            shutil.copy(self.predictor.report_path, save_path)
            QMessageBox.information(self, "成功", f"报告已保存至: {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存报告失败: {str(e)}")

    # 关闭事件
    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, '退出', "是否确定退出程序？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()


# 应用启动
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())