# -*- coding: utf-8 -*-
# @File : tiny_csp_darknet.py
# @Author: Runist
# @Time : 2020/8/10 11:41
# @Software: PyCharm
# @Brief: tiny-yolo4的backbone
from nets.yolo4 import DarknetConv2D_BN_Leaky, Conv2D_Upsample, yolo_feat_reshape
import config.config as cfg

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]


def tiny_resblock_body(inputs, num_filter):
    """
    残差块
    DarknetConv2D_BN_Leaky + 1次 darknet_block
    :param inputs: 上一层输出
    :param num_filter: conv的卷积核个数，每次残差块是不一样的
    :return: 卷积结果
    """
    # 特征整合
    x = DarknetConv2D_BN_Leaky(inputs, num_filter, 3)
    # 生成一个大的残差边(对应左边的shortcut)
    outer_shortconv = x

    # 通道分割(对应右边的卷积)
    # 这里的分割是把输入进来的特征层的最后一维度(通道数)进行分割，把最后一层分割成等同大小的两部分，取第二部分(为什么只要第二部分呢？)
    x = layers.Lambda(route_group, arguments={'groups': 2, 'group_id': 1})(x)
    x = DarknetConv2D_BN_Leaky(x, num_filter//2, 3)

    # 1次残差块
    # 内部的残差边inner_shortconv
    inner_shortconv = x
    x = DarknetConv2D_BN_Leaky(x, num_filter//2, 3)
    # 堆叠 - 两个特征层通道数都是 num_filter//2，堆叠完之后通道数变成num_filter
    x = layers.Concatenate()([x, inner_shortconv])
    # 进行通道整合 - 将通道数变为num_filter
    x = DarknetConv2D_BN_Leaky(x, num_filter, 1)

    # 第三个tiny_resblock_body会引出来一个有效特征层分支
    feat = x

    # 堆叠 - 两个特征层通道数都是 num_filter，堆叠之后通道数变成2*num_filter
    x = layers.Concatenate()([outer_shortconv, x])
    # 压缩特征层的高和宽
    x = layers.MaxPooling2D(pool_size=[2, 2])(x)

    # 最后对通道数进行整合
    return x, feat


def tiny_darknet_body(inputs):
    """
    tiny_darknet53是yolov4的特征提取网络，输出2个大小的特征层
    :param inputs: 输入图片[n, 416, 416, 3]
    :return:
    """
    # (416, 416, 3) -> (208, 208, 32)
    x = layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
    x = DarknetConv2D_BN_Leaky(x, 32, 3, strides=2)

    # (208, 208, 32) -> (104, 104, 64)
    x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(x, 64, 3, strides=2)

    # (104, 104, 64) -> (52, 52, 128)
    x, _ = tiny_resblock_body(x, 64)
    # (52, 52, 128) -> (26, 26, 256)
    x, _ = tiny_resblock_body(x, 128)
    # (26, 26, 256) -> (13, 13, 512)
    x, feat1 = tiny_resblock_body(x, 256)

    feat2 = DarknetConv2D_BN_Leaky(x, 512, 3)

    return feat1, feat2


def tiny_yolo4_body(input_shape):
    input_image = layers.Input(shape=input_shape, dtype='float32', name="input_1")  # [b, 416, 416, 3]

    # 生成darknet53的主干模型
    # 首先我们会获取到两个有效特征层,分别是
    # feat1 (26, 26, 256)
    # feat2 (13, 13, 512)
    feat1, feat2 = tiny_darknet_body(input_image)

    # (13, 13, 512) -> (13, 13, 256)
    y13 = DarknetConv2D_BN_Leaky(feat2, 256, kernel_size=1)
    output_13x13 = DarknetConv2D_BN_Leaky(y13, 512, kernel_size=3)

    # FPN特征融合
    output_13x13 = DarknetConv2D_BN_Leaky(output_13x13, len(cfg.anchor_masks[0]) * (cfg.num_classes+5), kernel_size=1, bn=False)

    # Conv2D + UpSampling2D (13, 13, 256) -> (13, 13, 128)
    y13_upsample = Conv2D_Upsample(y13, 128)

    # (26, 26, (128+256))
    y26 = layers.Concatenate()([feat1, y13_upsample])

    y26 = DarknetConv2D_BN_Leaky(y26, 256, kernel_size=3)
    output_26x26 = DarknetConv2D_BN_Leaky(y26, len(cfg.anchor_masks[0]) * (cfg.num_classes+5), kernel_size=1, bn=False)

    # 这里output1、output2的shape分别是 26x26, 13x13
    # 然后reshape为 从(b, size, size, 75) -> (b, size, size, 3, 25)
    output_26x26 = layers.Lambda(lambda x: yolo_feat_reshape(x), name='reshape_2')(output_26x26)
    output_13x13 = layers.Lambda(lambda x: yolo_feat_reshape(x), name='reshape_1')(output_13x13)

    return models.Model(input_image, [output_13x13, output_26x26])
