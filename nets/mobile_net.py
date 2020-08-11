# -*- coding: utf-8 -*-
# @File : mobile_net.py
# @Author: Runist
# @Time : 2020/8/4 15:30
# @Software: PyCharm
# @Brief:
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model, layers, callbacks, losses, optimizers, regularizers, metrics
from core.dataReader import ReadYolo4Data
import config.config as cfg
from tqdm import tqdm


def conv_bn_relu(inputs, out_channel, kernel_size=3, strides=(1, 1), padding='valid', name=''):
    """
    Conv2D + BN + ReLU
    :param inputs: 输入特征层
    :param out_channel: 输出特征层
    :param kernel_size: 卷积核大小
    :param strides: 步长
    :param padding: 补全的方式
    :param name: 层的名字前缀
    :return: 输出特征层
    """

    if "block_id" not in conv_bn_relu.__dict__:
        conv_bn_relu.block_id = 0
    conv_bn_relu.block_id += 1

    # 1x1的卷积 - 升高维度
    x = layers.Conv2D(filters=out_channel,
                      kernel_size=kernel_size,
                      padding=padding,
                      strides=strides,
                      use_bias=False,
                      name=name + "Conv2D_%d" % conv_bn_relu.block_id)(inputs)

    x = layers.BatchNormalization(name=name + 'BN_%d' % conv_bn_relu.block_id)(x)

    x = layers.ReLU(max_value=6.0, name=name + 'ReLU_%d' % conv_bn_relu.block_id)(x)

    return x


def invertedResidual(inputs, in_channel, out_channel, stride, expand_ratio):
    """
    倒残差结构
    :param inputs: 输入特征层
    :param in_channel: 输入通道数
    :param out_channel: 输出通道数
    :param stride: 步长
    :param expand_ratio: 倍乘因子
    :return: 输出特征层
    """
    if "block_id" not in invertedResidual.__dict__:
        invertedResidual.block_id = 0
    invertedResidual.block_id += 1

    # 倍乘率是决定中间的倒残差结构的通道数
    hidden_channel = in_channel * expand_ratio
    prefix = "Block_{}_".format(invertedResidual.block_id)

    if expand_ratio != 1:
        x = conv_bn_relu(inputs, hidden_channel, kernel_size=1, padding='same', name=prefix + "expand_")
    else:
        x = inputs

    if stride == 2:
        x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name=prefix + 'zero_pad')(x)

    # 3x3 depthwise conv
    x = layers.DepthwiseConv2D(kernel_size=3,
                               padding="same" if stride == 1 else 'valid',
                               strides=stride,
                               use_bias=False,
                               name=prefix + 'depthwise_Conv2D')(x)
    x = layers.BatchNormalization(name=prefix + "depthwise_BN")(x)
    x = layers.ReLU(6.0, name=prefix + "depthwise_ReLU")(x)

    # 1x1 pointwise conv(linear)
    x = layers.Conv2D(filters=out_channel,
                      kernel_size=1,
                      strides=1,
                      padding="SAME",
                      use_bias=False,
                      name=prefix + "pointwise_Conv2D")(x)
    x = layers.BatchNormalization(name=prefix + "pointwise_BN")(x)

    # 满足两个条件才能使用short cut
    if stride == 1 and in_channel == out_channel:
        return layers.Add(name=prefix + "add")([inputs, x])

    return x


def MobileNetV2(width, height, channel, num_classes, include_top=False):
    """
    MobileNetV2版本，相比MobileNetV1是在每个DW卷积、PW卷积之前多了一个通道递增的卷积，
    然后再将卷积后的结果与输入融合所以看起来是中间高，两边低
    :param width: 输入的宽度
    :param height: 输入的高度
    :param num_classes: 分类的类别数
    :param include_top: 是否包含输出层
    :return: model
    """
    input_image = layers.Input(shape=(width, height, channel), dtype="float32")

    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)), name='expand_Conv2D_pad_1')(input_image)
    x = conv_bn_relu(x, 32, kernel_size=3, strides=(2, 2), padding='valid')

    x = invertedResidual(x, x.shape[-1], 16, 1, 1)
    # ---2--- 第一层步距按照表格来,其他层步距为1
    x = invertedResidual(x, x.shape[-1], 32, 2, 6)
    x = invertedResidual(x, x.shape[-1], 32, 1, 6)
    # ---3---
    x = invertedResidual(x, x.shape[-1], 64, 2, 6)
    x = invertedResidual(x, x.shape[-1], 64, 1, 6)
    # ---4---
    x = invertedResidual(x, x.shape[-1], 128, 2, 6)
    x = invertedResidual(x, x.shape[-1], 128, 1, 6)
    # ---5---
    x = invertedResidual(x, x.shape[-1], 256, 1, 6)
    x = invertedResidual(x, x.shape[-1], 256, 1, 6)
    feat_52x52 = x
    # ---6---
    x = invertedResidual(x, x.shape[-1], 512, 1, 2)
    x = invertedResidual(x, x.shape[-1], 512, 1, 2)
    feat_26x26 = x
    # ---7---
    x = invertedResidual(x, x.shape[-1], 1024, 1, 2)
    x = invertedResidual(x, x.shape[-1], 1024, 1, 2)
    feat_13x13 = x

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)  # pool + flatten
        x = layers.Dense(num_classes)(x)

    model = Model(input_image, x)
    model.compile(loss=losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=optimizer,
                  metrics=["accuracy"])

    return model


@tf.function
def train_step(images, labels, model):
    """
    这里使用相对底层一点的方式进行训练
    :param images: 图片
    :param labels: 标签
    :return:
    """
    with tf.GradientTape() as tape:
        output = model(images, training=True)
        loss = loss_object(labels, output)

    # 反向传播梯度下降
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, output)


@tf.function
def test_step(images, labels, model):
    """
    测试，辅助分类器只有Output起作用，所以只用收集一个变量
    :param images:
    :param labels:
    :return:
    """
    output = model(images, training=False)
    t_loss = loss_object(labels, output)

    test_loss(t_loss)
    test_accuracy(labels, output)


class ClassifierDataRead(ReadYolo4Data):
    def _get_data(self, annotation_line, num_classes=2):
        line = str(annotation_line.numpy(), encoding="utf-8").split()
        # line = annotation_line.split()

        image_path = line[0]
        bbox = [list(map(int, box.split(','))) for box in line[1:]]
        # 这里只要标签信息，统计出现次数最多的分类，作为图片的分类
        bbox = [box[4] for box in bbox]

        bbox_set = list(set(bbox))
        label = bbox_set[0]
        max_freq = 0

        for i in set(bbox):
            temp = bbox.count(i)

            if temp > max_freq:
                label = i
            # elif temp == label:
            #     return
        label = tf.one_hot(label, depth=num_classes)

        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)

        image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
        input_height, input_width = self.input_shape

        image_height_f = tf.cast(image_height, tf.float32)
        image_width_f = tf.cast(image_width, tf.float32)
        input_height_f = tf.cast(input_height, tf.float32)
        input_width_f = tf.cast(input_width, tf.float32)

        scale = min(input_width_f / image_width_f, input_height_f / image_height_f)
        new_height = image_height_f * scale
        new_width = image_width_f * scale

        # 将图片按照固定长宽比进行缩放 空缺部分 padding
        dx_f = (input_width - new_width) / 2
        dy_f = (input_height - new_height) / 2
        dx = tf.cast(dx_f, tf.int32)
        dy = tf.cast(dy_f, tf.int32)

        # 其实这一块不是双三次线性插值resize导致像素点放大255倍，原因是：无论是cv还是plt在面对浮点数时，仅解释0-1完整比例
        image = tf.image.resize(image, [new_height, new_width], method=tf.image.ResizeMethod.BICUBIC)
        new_image = tf.image.pad_to_bounding_box(image, dy, dx, input_height, input_width)

        # 生成image.shape的大小的全1矩阵
        image_ones = tf.ones_like(image)
        image_ones_padded = tf.image.pad_to_bounding_box(image_ones, dy, dx, input_height, input_width)
        # 做个运算，白色区域变成0，填充0的区域变成1，再* 128，然后加上原图，就完成填充灰色的操作
        image = (1 - image_ones_padded) * 128 + new_image

        # 将图片归一化到0和1之间
        image /= 255.
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

        return image, label

    def parse(self, annotation_line):
        """
        为tf.data.Dataset.map编写合适的解析函数，由于函数中某些操作不支持
        python类型的操作，所以需要用py_function转换，定义的格式如下
            Args:
              @param annotation_line: 是一行数据（图片路径 + 预测框位置）
        tf.py_function
            Args:
              第一个是要转换成tf格式的python函数，
              第二个输入的参数，
              第三个是输出的类型
        """
        if cfg.data_pretreatment == "random":
            # 先对图片进行尺度处理，再对box位置处理成yolov4的格式
            image, label = tf.py_function(self._get_random_data, [annotation_line], [tf.float32, tf.float32])
        else:
            image, label = tf.py_function(self._get_data, [annotation_line], [tf.float32, tf.float32])

        h, w = self.input_shape

        image.set_shape([h, w, 3])

        # 如果py_function的输出有个[..., ...]那么结果也会是列表，一般单个使用的时候，可以不用加[]
        return image, label

    def make_datasets(self, annotation, mode="train"):
        """
        用tf.data的方式读取数据，以提高gpu使用率
        :param annotation: 数据行[image_path, [x,y,w,h,class ...]]
        :param mode: 训练集or验证集tf.data运行一次
        :return: 数据集
        """
        # 这是GPU读取方式
        dataset = tf.data.Dataset.from_tensor_slices(annotation)

        if mode == "train":
            # map的作用就是根据定义的 函数，对整个数据集都进行这样的操作
            # 而不用自己写一个for循环，如：可以自己定义一个归一化操作，然后用.map方法都归一化
            dataset = dataset.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # 打乱数据，这里的shuffle的值越接近整个数据集的大小，越贴近概率分布
            # 但是电脑往往没有这么大的内存，所以适量就好
            dataset = dataset.repeat().shuffle(buffer_size=cfg.shuffle_size).batch(self.batch_size)
            # prefetch解耦了 数据产生的时间 和 数据消耗的时间
            # prefetch官方的说法是可以在gpu训练模型的同时提前预处理下一批数据
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        else:
            # 验证集数据不需要增强
            cfg.data_pretreatment = 'normal'
            dataset = dataset.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.repeat().batch(self.batch_size).prefetch(self.batch_size)

        return dataset


def train_by_fit(model, train_datasets, valid_datasets, epochs, train_steps, valid_steps):
    best_test_loss = float('inf')
    train_datasets = iter(train_datasets)
    valid_datasets = iter(valid_datasets)

    # 创建summary
    summary_writer = tf.summary.create_file_writer(logdir=cfg.log_dir)

    for epoch in range(1, epochs + 1):
        train_loss.reset_states()       # clear history info
        train_accuracy.reset_states()   # clear history info
        test_loss.reset_states()        # clear history info
        test_accuracy.reset_states()    # clear history info

        process_bar = tqdm(range(train_steps), ncols=100, desc="Epoch {}".format(epoch), unit="step")
        for _ in process_bar:
            images, labels = next(train_datasets)
            train_step(images, labels, model)
            process_bar.set_postfix({'train_loss': '{:.5f}'.format(train_loss.result()),
                                     'train_acc': '{:.5f}'.format(train_accuracy.result())})

        process_bar = tqdm(range(valid_steps), ncols=100, desc="Epoch {}".format(epoch), unit="step")
        for _ in process_bar:
            images, labels = next(valid_datasets)
            test_step(images, labels, model)
            process_bar.set_postfix({'val_loss': '{:.5f}'.format(test_loss.result()),
                                     'val_acc': '{:.5f}'.format(test_accuracy.result())})

        template = 'Epoch {}, Loss: {:.2f}, Accuracy: {:.2f}, Test Loss: {:.2f}, Test Accuracy: {:.2f}\n'
        print(template.format(epoch,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))

        # 保存到tensorboard里
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=optimizer.iterations)
            tf.summary.scalar('validation_loss', test_loss.result(), step=optimizer.iterations)
            tf.summary.scalar('train_accuracy', train_accuracy.result(), step=optimizer.iterations)
            tf.summary.scalar('validation_accuracy', test_accuracy.result(), step=optimizer.iterations)

        if test_loss.result() < best_test_loss:
            best_test_loss = test_loss.result()
            model.save_weights("../logs/model/mobile_net.h5")


if __name__ == '__main__':
    epochs = 50
    batch_size = 2
    lr = 0.0001

    # 自定义损失、优化器、准确率
    loss_object = losses.CategoricalCrossentropy(from_logits=False)
    optimizer = optimizers.Adam(learning_rate=lr)

    train_loss = metrics.Mean(name='train_loss')
    train_accuracy = metrics.CategoricalAccuracy(name='train_accuracy')

    # 自定义损失和准确率方法
    test_loss = metrics.Mean(name='test_loss')
    test_accuracy = metrics.CategoricalAccuracy(name='test_accuracy')

    cfg.data_pretreatment = 'normal'
    reader = ClassifierDataRead("../config/train.txt", cfg.input_shape, batch_size)
    train_path, valid_path = reader.read_data_and_split_data()
    train_datasets = reader.make_datasets(train_path, "train")
    valid_datasets = reader.make_datasets(valid_path, "valid")

    train_steps = len(train_path) // batch_size
    valid_steps = len(valid_path) // batch_size

    model = MobileNetV2(416, 416, 3, num_classes=2, include_top=True)
    train_by_fit(model, train_datasets, valid_datasets, epochs, train_steps, valid_steps)

