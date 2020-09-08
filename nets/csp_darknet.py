# -*- coding: utf-8 -*-
# @File : csp_darknet.py
# @Author: Runist
# @Time : 2020/8/10 11:39
# @Software: PyCharm
# @Brief: yolo4的backbone
from nets.yolo4 import *
from tensorflow.keras import layers, models
import config.config as cfg


def resblock_body(inputs, filters, num_blocks, all_narrow=True):
    """
    残差块
    ZeroPadding + conv + nums_filters 次 darknet_block
    :param inputs: 上一层输出
    :param filters: conv的卷积核个数，每次残差块是不一样的
    :param num_blocks: 有几个这样的残差块
    :param all_narrow:
    :return: 卷积结果
    """
    # 进行长和宽的压缩(减半，这一部分和原本的Darknet53一样)
    preconv1 = layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
    preconv1 = DarknetConv2D_BN_Mish(preconv1, filters, kernel_size=3, strides=(2, 2))

    # 生成一个大的残差边(对应左边的shortcut)
    shortconv = DarknetConv2D_BN_Mish(preconv1, filters//2 if all_narrow else filters, kernel_size=1)

    # 主干部分的卷积(对应右边的卷积)
    mainconv = DarknetConv2D_BN_Mish(preconv1, filters//2 if all_narrow else filters, kernel_size=1)
    # 1x1卷积对通道数进行整合->3x3卷积提取特征，使用残差结构
    for i in range(num_blocks):
        x = DarknetConv2D_BN_Mish(mainconv, filters//2, kernel_size=1)
        x = DarknetConv2D_BN_Mish(x, filters//2 if all_narrow else filters, kernel_size=3)

        mainconv = layers.Add()([mainconv, x])

    # 1x1卷积后和残差边堆叠
    postconv = DarknetConv2D_BN_Mish(mainconv, filters//2 if all_narrow else filters, kernel_size=1)
    route = layers.Concatenate()([postconv, shortconv])

    # 最后对通道数进行整合
    output = DarknetConv2D_BN_Mish(route, filters, (1, 1))
    return output


def darknet_body(inputs):
    """
    darknet53是yolov4的特征提取网络，输出三个大小的特征层
    :param inputs: 输入图片[n, 416, 416, 3]
    :return:
    """
    x = DarknetConv2D_BN_Mish(inputs, 32, 3)
    x = resblock_body(x, 64, 1, False)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    feat_52x52 = x

    x = resblock_body(x, 512, 8)
    feat_26x26 = x

    x = resblock_body(x, 1024, 4)
    feat_13x13 = x

    return feat_52x52, feat_26x26, feat_13x13


def yolo4_body(input_shape):
    """
    yolov4的骨干网络部分，这里注释写的是输入情况是416x416的前提，如果输入大小时608x608的情况
    13x13 = 19x19、 26x26 = 38x38、 52x52 = 76x76
    对比yolov3（只有特征金字塔结构），在特征提取部分使用了PANet（下采样融合）的网络结构，加强了特征融合，提取更有效地特征
    :return: model
    """
    input_image = layers.Input(shape=input_shape, dtype='float32', name="input_1")  # [b, 416, 416, 3]
    if cfg.pretrain:
        print('Load weights {}.'.format(cfg.pretrain_weights_path))
        # 加载模型
        pretrain_model = tf.keras.models.load_model(cfg.pretrain_weights_path,
                                                    custom_objects={'Mish': Mish},
                                                    compile=False,)
        pretrain_model.summary()
        pretrain_model.trainable = False
        input_image = pretrain_model.input
        # feat_52x52, feat_26x26, feat_13x13 = pretrain_model.layers[131].output, \
        #                                      pretrain_model.layers[204].output, \
        #                                      pretrain_model.layers[247].output
        feat_52x52, feat_26x26, feat_13x13 = pretrain_model.get_layer("mish_37").output, \
                                             pretrain_model.get_layer("mish_58").output, \
                                             pretrain_model.get_layer("mish_71").output
    else:
        feat_52x52, feat_26x26, feat_13x13 = darknet_body(input_image)

    # 13x13 head
    # 三次卷积 + SPP + 三次卷积
    y13 = DarknetConv2D_BN_Leaky(feat_13x13, 512, kernel_size=1)
    y13 = DarknetConv2D_BN_Leaky(y13, 1024, kernel_size=3)
    y13 = DarknetConv2D_BN_Leaky(y13, 512, kernel_size=1)
    y13 = SPP_net(y13)
    y13 = DarknetConv2D_BN_Leaky(y13, 512, kernel_size=1)
    y13 = DarknetConv2D_BN_Leaky(y13, 1024, kernel_size=3)
    y13 = DarknetConv2D_BN_Leaky(y13, 512, kernel_size=1)
    y13_upsample = Conv2D_Upsample(y13, 256)

    # PANet
    # 26x26 head
    y26 = DarknetConv2D_BN_Leaky(feat_26x26, 256, kernel_size=1)
    y26 = layers.Concatenate(name="concatenate_6")([y26, y13_upsample])
    y26, _ = make_last_layers(y26, 256)     # TODO 到时候裁剪模型时就要把这里改一下
    y26_upsample = Conv2D_Upsample(y26, 128)

    # 52x52 head and output
    y52 = DarknetConv2D_BN_Leaky(feat_52x52, 128, (1, 1))
    y52 = layers.Concatenate(name="concatenate_7")([y52, y26_upsample])
    y52, output_52x52 = make_last_layers(y52, 128)

    # 26x26 output
    y52_downsample = layers.ZeroPadding2D(((1, 0), (1, 0)), name="zero_padding2d_5")(y52)
    y52_downsample = DarknetConv2D_BN_Leaky(y52_downsample, 256, kernel_size=3, strides=(2, 2))
    y26 = layers.Concatenate(name="concatenate_8")([y52_downsample, y26])
    y26, output_26x26 = make_last_layers(y26, 256)

    # 13x13 output
    y26_downsample = layers.ZeroPadding2D(((1, 0), (1, 0)), name="zero_padding2d_6")(y26)
    y26_downsample = DarknetConv2D_BN_Leaky(y26_downsample, 512, kernel_size=3, strides=(2, 2))
    y13 = layers.Concatenate(name="concatenate_9")([y26_downsample, y13])
    y13, output_13x13 = make_last_layers(y13, 512)

    # 这里output1、output2、output3的shape分别是52x52, 26x26, 13x13
    # 然后reshape为 从(b, size, size, 75) -> (b, size, size, 3, 25)
    output_52x52 = layers.Lambda(lambda x: yolo_feat_reshape(x), name='reshape_3')(output_52x52)
    output_26x26 = layers.Lambda(lambda x: yolo_feat_reshape(x), name='reshape_2')(output_26x26)
    output_13x13 = layers.Lambda(lambda x: yolo_feat_reshape(x), name='reshape_1')(output_13x13)

    return models.Model(input_image, [output_13x13, output_26x26, output_52x52])
