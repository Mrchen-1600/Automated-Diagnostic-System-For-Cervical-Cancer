本目录下文件说明：
    SplitDtaSet.py:
        用于数据集的拆分，按照6：2：2比例划分成：训练集、验证集和测试集
    RenameImage.py:
        用于给测试集图片重命名，在细胞图像的文件名上加上其正确的类别名，便于网络预测时统计预测精度。
    RemoveBlackEdge.py:
        在数据增强时，如果选用旋转的增强方法，可能导致大量有效像素丟失，引入过多无信息的黑边，
        这些图像的增强质量不高，对网络训练并无优势，该程序用于删除这些增强效果不佳的图像。
    MoveImages.py:
        用于将测试集所有存放在子目录下的图像移动到测试集目录下。