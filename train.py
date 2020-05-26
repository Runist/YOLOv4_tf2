# -*- coding: utf-8 -*-
# @File : train.py
# @Author: Runist
# @Time : 2020/5/21 17:11
# @Software: PyCharm
# @Brief:


from nets.loss import YoloLoss
from nets.model import yolo4_body
import config.config as cfg
from core.dataReader import ReadYolo4Data

import os
import shutil
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint


def main():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # 读取数据
    reader = ReadYolo4Data(cfg.annotation_path, cfg.input_shape, cfg.batch_size)
    train, valid = reader.read_data_and_split_data()
    train_datasets = reader.make_datasets(train, "train")
    valid_datasets = reader.make_datasets(valid, "valid")
    train_steps = len(train) // cfg.batch_size
    valid_steps = len(valid) // cfg.batch_size

    if os.path.exists(cfg.log_dir):
        # 清除summary目录下原有的东西
        for f in os.listdir(cfg.log_dir):
            file = os.path.join(cfg.log_dir, f)
            shutil.rmtree(file)

    # 建立模型保存目录
    if not os.path.exists(os.path.split(cfg.model_path)[0]):
        os.mkdir(os.path.split(cfg.model_path)[0])

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train), len(valid), cfg.batch_size))
    if cfg.train_mode == "eager":
        pass
    else:
        # 创建summary
        writer = tf.summary.create_file_writer(logdir=cfg.log_dir + '/loss')

        optimizer = Adam(learning_rate=cfg.learn_rating)
        yolo_loss = [YoloLoss(cfg.anchors[mask],
                              label_smoothing=cfg.label_smoothing,
                              summary_writer=writer,
                              optimizer=optimizer) for mask in cfg.anchor_masks]

        train_by_fit(optimizer, yolo_loss, train_datasets, valid_datasets, train_steps, valid_steps)


def create_model():
    """
    创建模型，方便MirroredStrategy的操作
    :return: Model
    """
    # 是否预训练
    if cfg.pretrain:
        print('Load weights {}.'.format(cfg.pretrain_weights_path))
        # 定义模型
        pretrain_model = tf.keras.models.load_model(cfg.pretrain_weights_path, compile=False)
        pretrain_model.trainable = False
        input_image = pretrain_model.input
        feat_52x52, feat_26x26, feat_13x13 = pretrain_model.layers[92].output, \
                                             pretrain_model.layers[152].output, \
                                             pretrain_model.layers[184].output
        model = yolo4_body([input_image, feat_52x52, feat_26x26, feat_13x13])
    else:
        print("Train all layers.")
        model = yolo4_body()

    return model


def train_by_fit(optimizer, loss, train_datasets, valid_datasets, train_steps, valid_steps):
    """
    使用fit方式训练，可以知道训练完的时间，以及更规范的添加callbacks参数
    :param optimizer: 优化器
    :param loss: 自定义的loss function
    :param train_datasets: 以tf.data封装好的训练集数据
    :param valid_datasets: 验证集数据
    :param train_steps: 迭代一个epoch的轮次
    :param valid_steps: 同上
    :return: None
    """
    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=10, verbose=1),
        TensorBoard(log_dir=cfg.log_dir),
        ModelCheckpoint(cfg.log_dir, save_best_only=True, save_weights_only=True)
    ]

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = create_model()
        model.compile(optimizer=optimizer, loss=loss)

    # initial_epoch用于恢复之前的训练
    model.fit(train_datasets,
              steps_per_epoch=max(1, train_steps),
              validation_data=valid_datasets,
              validation_steps=max(1, valid_steps),
              epochs=cfg.epochs,
              initial_epoch=0,
              callbacks=callbacks)

    model.save_weights(cfg.model_path)

    if cfg.fine_tune:
        with strategy.scope():
            print("Unfreeze all of the layers.")
            for i in range(len(model.layers)):
                model.layers[i].trainable = True

            model.compile(optimizer=Adam(learning_rate=cfg.learn_rating / 10), loss=loss)

        model.fit(train_datasets,
                  steps_per_epoch=max(1, train_steps),
                  validation_data=valid_datasets,
                  validation_steps=max(1, valid_steps),
                  epochs=cfg.epochs*2,
                  initial_epoch=cfg.epochs + 1,
                  callbacks=callbacks)

        model.save_weights(cfg.model_path)


if __name__ == '__main__':
    main()
