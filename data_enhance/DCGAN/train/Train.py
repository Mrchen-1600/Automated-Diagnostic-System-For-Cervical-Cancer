# -*- coding:utf-8 -*-
"""
作者：尘小风
软件：Pycharm2020.2
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, utils
from torch.utils.data import DataLoader
from data_enhance.DCGAN.config.Config import opt
from data_enhance.DCGAN.model.Model import Generator, Discriminator
import os

def train():
    # 检查并创建保存目录
    os.makedirs(opt.save_path, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 修复了缺少括号的错误
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.CenterCrop(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        dataset = datasets.ImageFolder(root=opt.data_path, transform=transform)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=opt.batch_size,
                                shuffle=True,
                                num_workers=opt.num_workers,
                                drop_last=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    gen = Generator(opt).to(device)
    dis = Discriminator(opt).to(device)

    g_optim = optim.Adam(gen.parameters(), lr=opt.lr1, betas=(opt.beta1, 0.99))
    d_optim = optim.Adam(dis.parameters(), lr=opt.lr2, betas=(opt.beta1, 0.99))
    loss_function = nn.BCELoss().to(device)

    true_labels = torch.ones(opt.batch_size).to(device)
    fake_labels = torch.zeros(opt.batch_size).to(device)
    noises = torch.randn(opt.batch_size, opt.nz, 1, 1).to(device)
    fix_noises = torch.randn(1, opt.nz, 1, 1).to(device)

    for epoch in range(opt.max_epoch):
        for step, (img, _) in enumerate(dataloader):
            try:
                real_img = img.to(device)

                # 训练判别器
                if step % opt.d_every == 0:
                    d_optim.zero_grad()

                    real_output = dis(real_img)
                    d_real_loss = loss_function(real_output, true_labels)
                    d_real_loss.backward()

                    gen_img = gen(noises)
                    fake_output = dis(gen_img.detach())
                    d_fake_loss = loss_function(fake_output, fake_labels)
                    d_fake_loss.backward()

                    d_optim.step()

                # 训练生成器
                if step % opt.g_every == 0:
                    g_optim.zero_grad()
                    noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
                    fake_img = gen(noises)
                    fake_output = dis(fake_img)
                    g_loss = loss_function(fake_output, true_labels)
                    g_loss.backward()
                    g_optim.step()

            except Exception as e:
                print(f"Error in epoch {epoch} step {step}: {e}")
                continue

        # 保存模型和生成样本
        if (epoch + 1) % opt.save_every == 0:
            try:
                print(f'Epoch: {epoch + 1}/{opt.max_epoch}')

                # 生成固定噪声的样本用于观察训练进展
                with torch.no_grad():
                    fix_fake_image = gen(fix_noises)

                # 文件保存
                save_filename = os.path.join(opt.save_path, f"{epoch + 1}.bmp")
                utils.save_image(
                    fix_fake_image.data,
                    save_filename,
                    normalize=True,
                    nrow=1
                )

                # 保存模型
                torch.save(dis.state_dict(), f'checkpoints/dis_{epoch + 1}.pth')
                torch.save(gen.state_dict(), f'checkpoints/gen_{epoch + 1}.pth')

            except Exception as e:
                print(f"Error saving results at epoch {epoch + 1}: {e}")


if __name__ == '__main__':
    if not hasattr(opt, 'save_path'):
        opt.save_path = '../enhance_imgs/'

    train()