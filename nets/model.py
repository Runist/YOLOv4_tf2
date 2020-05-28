# -*- coding: utf-8 -*-
# @File : model.py
# @Author: Runist
# @Time : 2020/5/20 16:03
# @Software: PyCharm
# @Brief: yolov4的模型结构

from tensorflow.keras.layers import Input, Conv2D, Add, ZeroPadding2D, Layer, LeakyReLU, Lambda
from tensorflow.keras.layers import Concatenate, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import config.config as cfg


class Mish(Layer):
    """
    Mish激活函数
    公式：
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: 任意的. 使用参数 `input_shape`
        - Output: 和输入一样的shape
    Examples:
        >>> X_input = Input(input_shape)
        >>> X = Mish()(X_input)
    """
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    @staticmethod
    def call(inputs):
        return inputs * tf.tanh(tf.nn.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    @staticmethod
    def compute_output_shape(input_shape):
        return input_shape


def DarknetConv2D_BN_Mish(inputs, num_filter, kernel_size, strides=(1, 1), bn=True):
    """
    卷积 + 批归一化 + leaky激活，因为大量用到这样的结构，所以这样写
    :param inputs: 输入
    :param num_filter: 卷积个数
    :param kernel_size: 卷积核大小
    :param strides: 步长
    :param bn: 是否使用批归一化
    :return: x
    """
    if strides == (1, 1) or strides == 1:
        padding = 'same'
    else:
        padding = 'valid'

    x = Conv2D(num_filter, kernel_size=kernel_size,
               strides=strides, padding=padding,                        # 这里的参数是只l2求和之后所乘上的系数
               use_bias=not bn, kernel_regularizer=l2(5e-4))(inputs)    # 只有添加正则化参数，才能调用model.losses方法
    if bn:
        x = BatchNormalization()(x)
        x = Mish()(x)

    return x


def DarknetConv2D_BN_Leaky(inputs, num_filter, kernel_size, strides=(1, 1), bn=True):
    """
    卷积 + 批归一化 + leaky激活，因为大量用到这样的结构，
    其中名字的管理比较麻烦，所以添加了函数内部变量
    :param inputs: 输入
    :param num_filter: 卷积个数
    :param kernel_size: 卷积核大小
    :param strides: 步长
    :param bn: 是否使用批归一化
    :return: x
    """
    if "conv2d" not in DarknetConv2D_BN_Leaky.__dict__ and cfg.pretrain:
        DarknetConv2D_BN_Leaky.conv2d = 72

    if "bn" not in DarknetConv2D_BN_Leaky.__dict__ and cfg.pretrain:
        DarknetConv2D_BN_Leaky.bn = 72

    if strides == (1, 1) or strides == 1:
        padding = 'same'
    else:
        padding = 'valid'

    if cfg.pretrain:
        conv2d_name = "conv2d_{}".format(DarknetConv2D_BN_Leaky.conv2d)
        DarknetConv2D_BN_Leaky.conv2d += 1
    else:
        conv2d_name = None

    x = Conv2D(num_filter, kernel_size=kernel_size,
               strides=strides, padding=padding,                        # 这里的参数是只l2求和之后所乘上的系数
               use_bias=not bn, kernel_regularizer=l2(5e-4),            # 只有添加正则化参数，才能调用model.losses方法
               name=conv2d_name)(inputs)

    if bn:
        if cfg.pretrain:
            bn_name = "batch_normalization_{}".format(DarknetConv2D_BN_Leaky.bn)
            DarknetConv2D_BN_Leaky.bn += 1
        else:
            bn_name = None

        x = BatchNormalization(name=bn_name)(x)
        # alpha是x < 0时，变量系数
        x = LeakyReLU(alpha=0.1)(x)

    return x


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
    preconv1 = ZeroPadding2D(((1, 0), (1, 0)))(inputs)
    preconv1 = DarknetConv2D_BN_Mish(preconv1, filters, kernel_size=3, strides=(2, 2))

    # 生成一个大的残差边(对应左边的shortcut)
    shortconv = DarknetConv2D_BN_Mish(preconv1, filters//2 if all_narrow else filters, kernel_size=1)

    # 主干部分的卷积(对应右边的卷积)
    mainconv = DarknetConv2D_BN_Mish(preconv1, filters//2 if all_narrow else filters, kernel_size=1)
    # 1x1卷积对通道数进行整合->3x3卷积提取特征，使用残差结构
    for i in range(num_blocks):
        x = DarknetConv2D_BN_Mish(mainconv, filters//2, kernel_size=1)
        x = DarknetConv2D_BN_Mish(x, filters//2 if all_narrow else filters, kernel_size=3)

        mainconv = Add()([mainconv, x])

    # 1x1卷积后和残差边堆叠
    postconv = DarknetConv2D_BN_Mish(mainconv, filters//2 if all_narrow else filters, kernel_size=1)
    route = Concatenate()([postconv, shortconv])

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


def make_last_layers(inputs, num_filter):
    """
    5次（conv + bn + leaky激活）
    2次（conv + bn + leaky激活）
    :param inputs: 输入
    :param num_filter: 卷积核个数
    :return: x
    """
    x = DarknetConv2D_BN_Leaky(inputs, num_filter, kernel_size=1)
    x = DarknetConv2D_BN_Leaky(x, num_filter * 2, kernel_size=3)
    x = DarknetConv2D_BN_Leaky(x, num_filter, kernel_size=1)
    x = DarknetConv2D_BN_Leaky(x, num_filter * 2, kernel_size=3)
    output_5 = DarknetConv2D_BN_Leaky(x, num_filter, kernel_size=1)

    x = DarknetConv2D_BN_Leaky(output_5, num_filter * 2, kernel_size=3)
    output_7 = DarknetConv2D_BN_Leaky(x, cfg.num_bbox * (cfg.num_classes+5), kernel_size=1, bn=False)

    return output_5, output_7


def SPP_net(inputs):
    """
    SPP结构，使得图片可以是任意大小，但输出是一样的。
    对图片进行三次MaxPooling2D，得到不同的感受野，在进行堆叠。得到原来的通道数x4的输出层
    :param inputs:
    :return:
    """
    # 使用了SPP结构，即不同尺度的最大池化后堆叠。
    maxpool1 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(inputs)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(inputs)
    maxpool3 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(inputs)
    output = Concatenate(name="concatenate_5")([maxpool1, maxpool2, maxpool3, inputs])

    return output


def Conv2D_Upsample(inputs, num_filter):
    """
    1次（conv + bn + leaky激活） + 上采样
    :param inputs: 输入层
    :param num_filter: 卷积核个数
    :return: x
    """
    x = DarknetConv2D_BN_Leaky(inputs, num_filter, kernel_size=1)
    x = UpSampling2D(2)(x)

    return x


def yolo4_body():
    """
    yolov4的骨干网络部分，这里注释写的是输入情况是416x416的前提，如果输入大小时608x608的情况
    13x13 = 19x19、 26x26 = 38x38、 52x52 = 76x76
    对比yolov3（只有特征金字塔结构），在特征提取部分使用了PANet（下采样融合）的网络结构，加强了特征融合，提取更有效地特征
    :return:
    """
    height, width = cfg.input_shape
    input_image = Input(shape=(height, width, 3), dtype='float32', name="input_1")  # [b, 416, 416, 3]
    if cfg.pretrain:
        print('Load weights {}.'.format(cfg.pretrain_weights_path))
        # 加载模型
        pretrain_model = tf.keras.models.load_model(cfg.pretrain_weights_path,
                                                    custom_objects={'Mish': Mish},
                                                    compile=False,)
        pretrain_model.trainable = False
        input_image = pretrain_model.input
        # feat_52x52, feat_26x26, feat_13x13 = pretrain_model.layers[131].output, \
        #                                      pretrain_model.layers[204].output, \
        #                                      pretrain_model.layers[247].output
        feat_52x52, feat_26x26, feat_13x13 = pretrain_model.get_layer("mish_37").output, \
                                             pretrain_model.get_layer("mish_58").output, \
                                             pretrain_model.get_layer("mish_71").output
    else:
        print("Train all layers.")
        feat_52x52, feat_26x26, feat_13x13 = darknet_body(input_image)

    # 13x13 head
    # 三次卷积 + spp + 三次卷积
    y13 = DarknetConv2D_BN_Leaky(feat_13x13, 512, kernel_size=1)
    y13 = DarknetConv2D_BN_Leaky(y13, 1024, kernel_size=3)
    y13 = DarknetConv2D_BN_Leaky(y13, 512, kernel_size=1)
    y13 = SPP_net(y13)
    y13 = DarknetConv2D_BN_Leaky(y13, 512, kernel_size=1)
    y13 = DarknetConv2D_BN_Leaky(y13, 1024, kernel_size=3)
    y13 = DarknetConv2D_BN_Leaky(y13, 512, kernel_size=1)
    y13_upsample = Conv2D_Upsample(y13, 256)

    # 26x26 head
    y26 = DarknetConv2D_BN_Leaky(feat_26x26, 256, kernel_size=1)
    y26 = Concatenate(name="concatenate_6")([y26, y13_upsample])
    y26, _ = make_last_layers(y26, 256)     # TODO 到时候裁剪模型时就要把这里改一下
    y26_upsample = Conv2D_Upsample(y26, 128)

    # 52x52 head and output
    y52 = DarknetConv2D_BN_Leaky(feat_52x52, 128, (1, 1))
    y52 = Concatenate(name="concatenate_7")([y52, y26_upsample])
    y52, output_52x52 = make_last_layers(y52, 128)

    # 26x26 output
    y52_downsample = ZeroPadding2D(((1, 0), (1, 0)), name="zero_padding2d_5")(y52)
    y52_downsample = DarknetConv2D_BN_Leaky(y52_downsample, 256, kernel_size=3, strides=(2, 2))
    y26 = Concatenate(name="concatenate_8")([y52_downsample, y26])
    y26, output_26x26 = make_last_layers(y26, 256)

    # 13x13 output
    y26_downsample = ZeroPadding2D(((1, 0), (1, 0)), name="zero_padding2d_6")(y26)
    y26_downsample = DarknetConv2D_BN_Leaky(y26_downsample, 512, kernel_size=3, strides=(2, 2))
    y13 = Concatenate(name="concatenate_9")([y26_downsample, y13])
    y13, output_13x13 = make_last_layers(y13, 512)

    # 这里output1、output2、output3的shape分别是52x52, 26x26, 13x13
    # 然后reshape为 从(b, size, size, 75) -> (b, size, size, 3, 25)
    output_52x52 = Lambda(lambda x: yolo_feat_reshape(x), name='reshape_3')(output_52x52)
    output_26x26 = Lambda(lambda x: yolo_feat_reshape(x), name='reshape_2')(output_26x26)
    output_13x13 = Lambda(lambda x: yolo_feat_reshape(x), name='reshape_1')(output_13x13)

    model = Model(input_image, [output_13x13, output_26x26, output_52x52])

    return model


def yolo_feat_reshape(feat):
    """
    处理一下y_pred的数据，reshape，从b, 13, 13, 75 -> b, 13, 13, 3, 25
    在Keras.model编译前处理是为了loss计算上能匹配
    :param feat:
    :return:
    """
    grid_size = tf.shape(feat)[1]
    reshape_feat = tf.reshape(feat, [-1, grid_size, grid_size, cfg.num_bbox, cfg.num_classes + 5])

    return reshape_feat


def yolo4_head(y_pred, anchors, calc_loss=False):
    """
    处理一下y_pred的数据，reshape，从b, 13, 13, 75 -> b, 13, 13, 3, 25
    另外，取名为head是有意义的。因为目标检测大多数分为 - Backbone - Detection head两个部分
    :param y_pred: 预测数据
    :param anchors: 其中一种大小的先验框（总共三种）
    :param calc_loss: 是否计算loss，该函数可以在直接预测的地方用
    :return:
        bbox: 存储了x1, y1 x2, y2的坐标 shape(b, 13, 13 ,3, 4)
        objectness: 该分类的置信度 shape(b, 13, 13 ,3, 1)
        class_probs: 存储了20个分类在sigmoid函数激活后的数值 shape(b, 13, 13 ,3, 20)
        pred_xywh: 把xy(中心点),wh shape(b, 13, 13 ,3, 4)
    """
    grid_size = tf.shape(y_pred)[1]

    # tf.spilt的参数对应：2-(x,y) 2-(w,h) 1-置信度 classes=20-分类数目的得分
    box_xy, box_wh, confidence, class_probs = tf.split(y_pred, (2, 2, 1, cfg.num_classes), axis=-1)
    # 举例：box_xy (13, 13, 3, 2) 3是指三个框，2是xy，其他三个输出类似

    # sigmoid是为了让tx, ty在[0, 1]，防止偏移过多，使得中心点落在一个网络单元格中，这也是激活函数的作用（修正）
    # 而对confidence和class_probs使用sigmoid是为了得到0-1之间的概率
    box_xy = tf.sigmoid(box_xy)
    confidence = tf.sigmoid(confidence)
    class_probs = tf.sigmoid(class_probs)

    # !!! grid[x][y] == (y, x)
    # sigmoid(x) + cx，在这里看，生成grid的原因是要和y_true的格式对齐。
    # 而且加上特征图就是13x13 26x26...一个特征图上的点，就预测一个结果。
    grid_y = tf.tile(tf.reshape(tf.range(grid_size), [-1, 1, 1, 1]), [1, grid_size, 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(grid_size), [1, -1, 1, 1]), [grid_size, 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], axis=-1)  # [gx, gy, 1, 2]
    grid = tf.cast(grid, tf.float32)

    # 把xy, wh归一化成比例
    # box_xy(b, 13, 13, 3, 2)  grid(13, 13, 1, 2)  grid_size shape-()-13
    # box_wh(b, 13, 13, 3, 2)  anchors_tensor(1, 1, 1, 3, 2)
    box_xy = (box_xy + grid) / tf.cast(grid_size, tf.float32)
    # 要注意，xy除去的是13，wh除去的416，是因为下面wh用的也是416(如果xywh不归一化，和概率值一起训练肯定不收敛啊)
    box_wh = tf.exp(box_wh) * anchors / cfg.input_shape
    # 最后 box_xy、box_wh 都是 (b, 13, 13, 3, 2)

    # 把xy,wh 合并成pred_box在最后一个维度上（axis=-1）
    pred_xywh = tf.concat([box_xy, box_wh], axis=-1)  # original xywh for loss

    if calc_loss:
        return pred_xywh, grid

    return box_xy, box_wh, confidence, class_probs



