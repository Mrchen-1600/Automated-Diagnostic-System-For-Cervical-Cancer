# -*- coding:utf-8 -*-
"""
作者：尘小风
软件：Pycharm2020.2
"""

class Config(object):
    data_path = '../../../dataset/SIPaKMeD/cell/'
    num_workers = 0
    img_size = 96
    batch_size = 64
    max_epoch = 300
    lr1 = 0.0002
    lr2 = 0.0002
    beta1 = 0.5
    gpu = True
    nz = 100
    ngf = 64
    ndf = 64

    # save model parameter
    save_path = '../enhance_imgs/'
    d_every = 1
    g_every = 5
    save_every = 10
    netd_path = None
    netg_path = None

    gen_img = "result.png"
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0
    gen_std = 1

opt = Config()