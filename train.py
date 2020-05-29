# -*- coding: utf-8 -*-
# @File : train.py
# @Author: Runist
# @Time : 2020/5/21 17:11
# @Software: PyCharm
# @Brief:


from nets.loss import YoloLoss, WarmUpCosineDecayScheduler
from nets.model import yolo4_body, Mish
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
        train_by_fit(train_datasets, valid_datasets, train_steps, valid_steps)


def train_by_fit(train_datasets, valid_datasets, train_steps, valid_steps):
    """
    使用fit方式训练，可以知道训练完的时间，以及更规范的添加callbacks参数
    :param train_datasets: 以tf.data封装好的训练集数据
    :param valid_datasets: 验证集数据
    :param train_steps: 迭代一个epoch的轮次
    :param valid_steps: 同上
    :return: None
    """
    # 最大学习率
    learning_rate_base = cfg.learning_rate
    if cfg.cosine_scheduler:
        # 预热期
        warmup_epoch = int(cfg.epochs * 0.2)
        # 总共的步长
        total_steps = int(cfg.epochs * train_steps)
        # 预热步长
        warmup_steps = int(warmup_epoch * train_steps)
        # 学习率
        reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                               total_steps=total_steps,
                                               warmup_learning_rate=learning_rate_base/10,
                                               warmup_steps=warmup_steps,
                                               hold_base_rate_steps=train_steps,
                                               min_learn_rate=learning_rate_base/1000)
        optimizer = Adam()
    else:
        optimizer = Adam(learning_rate_base)

    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        TensorBoard(log_dir=cfg.log_dir),
        ModelCheckpoint(cfg.best_model, save_best_only=True, save_weights_only=True),
        reduce_lr if cfg.cosine_scheduler else ReduceLROnPlateau(verbose=1)
    ]

    # 创建summary，收集具体的loss信息
    writer = tf.summary.create_file_writer(logdir=cfg.log_dir + '/loss')
    yolo_loss = [YoloLoss(cfg.anchors[mask],
                          label_smoothing=cfg.label_smoothing,
                          summary_writer=writer,
                          optimizer=optimizer) for mask in cfg.anchor_masks]

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = yolo4_body()
        model.compile(optimizer=optimizer, loss=yolo_loss)

    model.fit(train_datasets,
              steps_per_epoch=max(1, train_steps),
              validation_data=valid_datasets,
              validation_steps=max(1, valid_steps),
              epochs=cfg.epochs,
              initial_epoch=0,
              callbacks=callbacks)

    model.save_weights(cfg.model_path)

    if cfg.fine_tune:
        cfg.batch_size = 2
        # 最大学习率
        learning_rate_base = cfg.learning_rate / 10

        if cfg.cosine_scheduler:
            # 预热期
            warmup_epoch = int(cfg.epochs * 0.2)
            # 总共的步长
            total_steps = int(cfg.epochs * train_steps)
            # 预热步长
            warmup_steps = int(warmup_epoch * train_steps)
            # 学习率
            reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                                   total_steps=total_steps,
                                                   warmup_learning_rate=learning_rate_base/10,
                                                   warmup_steps=warmup_steps,
                                                   hold_base_rate_steps=train_steps // 2,
                                                   min_learn_rate=learning_rate_base/100)
            optimizer = Adam()
        else:
            optimizer = Adam(learning_rate_base)

        callbacks = [
            EarlyStopping(patience=10, verbose=1),
            TensorBoard(log_dir=cfg.log_dir),
            ModelCheckpoint(cfg.best_model, save_best_only=True, save_weights_only=True),
            reduce_lr if cfg.cosine_scheduler else ReduceLROnPlateau(verbose=1)
        ]
        with strategy.scope():
            print("Unfreeze all of the layers.")
            for i in range(len(model.layers)):
                model.layers[i].trainable = True

            model.compile(optimizer=optimizer, loss=yolo_loss)

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
