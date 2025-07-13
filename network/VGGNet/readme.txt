本目录下文件说明：
    config: 目录下存放的是训练和预测的相关配置参数，在网络训练和预测之前，前需要先根据自己的需求修改Config.py文件中的参数
    model: 目录下存放的是VGGNet的模型结构
    train: 目录下存放的是训练的代码文件和预训练权重文件（需要自行下载，本代码中重命名成了vgg16_bn-pre.pth）
        官方的预训练权重文件地址：
        model_urls = {
            'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
            'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
            'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
            'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
        }
        训练后，在VGGNet目录下会多一个output文件夹，下面有两个子目录runs和weights，
            runs目录下存放的是训练过程中的日志文件，
                可通过在终端cd到output目录下，然后运行命令：tensorboard --logdir=runs --port=10111，('runs'是日志文件所在目录，'10111'是指定的访问端口号，只需要和其他程序占用的端口号不同即可)
                进入tensorboard可视化界面，查看训练过程中生成的准确率、损失和学习率曲线等图像
            weights目录下存放的是训练过程中保存的权重文件，会保存每一个epoch的权重文件，如果只需要保存准确率最高的权重文件，在训练文件中注释掉相应的权重保存代码即可
            同时，也会保存一份验证准确率最高的权重文件best_model.pth

    predict: 目录下存放的是预测的相关代码，使用训练好的权重文件best_model.pth进行批量预测

注：在预测之前，确保之前已经运行过util目录下的RenameImage.py和MoveImage.py脚本，
    RenameImage.py的作用是对测试集图片进行重命名，预测的准确率是通过预测的类别和图像名中带有的实际类别对比计算得到的，
    因此需要对文件进行重命名，使其图片名中带有实际类别
    MoveImage.py的作用是将测试集所有存放在子目录下的图片移动到测试集目录下，方便后续进行批量预测

