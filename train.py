# -*- coding: utf-8 -*-
# @File : train.py
# @Author: Runist
# @Time : 2020/5/21 17:11
# @Software: PyCharm
# @Brief: Yolov4训练启动脚本


from nets.loss import YoloLoss, WarmUpCosineDecayScheduler
from nets.model import yolo4_body
import config.config as cfg
from core.dataReader import ReadYolo4Data

import os
import shutil
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.metrics import Mean
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
            if isinstance(file, str):
                os.remove(file)
            else:
                shutil.rmtree(file)

    # 建立模型保存目录
    if not os.path.exists(os.path.split(cfg.model_path)[0]):
        os.mkdir(os.path.split(cfg.model_path)[0])

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train), len(valid), cfg.batch_size))
    if cfg.train_mode == "eager":
        train_by_eager(train_datasets, valid_datasets, train_steps, valid_steps)
    else:
        train_by_fit(train_datasets, valid_datasets, train_steps, valid_steps)


def train_by_eager(train_datasets, valid_datasets, train_steps, valid_steps):
    # 创建模型结构
    model = yolo4_body()

    # 定义模型评估指标
    train_loss = Mean(name='train_loss')
    valid_loss = Mean(name='valid_loss')

    # 设置保存最好模型的指标
    best_test_loss = float('inf')
    patience = 10
    min_delta = 1e-3
    patience_cnt = 0
    history_loss = []

    # 创建优化器
    # 定义优化器和学习率衰减速率
    # PolynomialDecay参数：cfg.learn_rating 经过 cfg.epochs 衰减到 cfg.learn_rating/10
    # 1、lr_fn是类似是一个函数，每次需要它来计算当前学习率都会调用它
    # 2、它具有一个内部计数器，每次调用apply_gradients，就会+1
    lr_fn = PolynomialDecay(cfg.learning_rate, cfg.epochs, cfg.learning_rate / 10, 2)
    optimizer = Adam(learning_rate=lr_fn)

    # 创建summary
    summary_writer = tf.summary.create_file_writer(logdir=cfg.log_dir)

    # 定义loss
    yolo_loss = [YoloLoss(cfg.anchors[mask],
                          label_smooth=cfg.label_smooth,
                          summary_writer=summary_writer,
                          optimizer=optimizer) for mask in cfg.anchor_masks]

    # low level的方式计算loss
    for epoch in range(1, cfg.epochs + 1):
        train_loss.reset_states()
        valid_loss.reset_states()
        step = 0
        print("Epoch {}/{}".format(epoch, cfg.epochs))

        # 处理训练集数据
        for batch, (images, labels) in enumerate(train_datasets.take(train_steps)):
            with tf.GradientTape() as tape:
                # 得到预测
                outputs = model(images, training=True)
                # 计算损失(注意这里收集model.losses的前提是Conv2D的kernel_regularizer参数)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                # yolo_loss、label、output都是3个特征层的数据，通过for 拆包之后，一个loss_fn就是yolo_loss中一个特征层
                # 然后逐一计算,
                for output, label, loss_fn in zip(outputs, labels, yolo_loss):
                    pred_loss.append(loss_fn(label, output))

                # 总损失 = yolo损失 + 正则化损失
                total_train_loss = tf.reduce_sum(pred_loss) + regularization_loss

            # 反向传播梯度下降
            # model.trainable_variables代表把loss反向传播到每个可以训练的变量中
            grads = tape.gradient(total_train_loss, model.trainable_variables)
            # 将每个节点的误差梯度gradients，用于更新该节点的可训练变量值
            # zip是把梯度和可训练变量值打包成元组
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # 更新train_loss
            train_loss.update_state(total_train_loss)
            # 输出训练过程
            rate = (step + 1) / train_steps
            a = "=" * int(rate * 30)
            b = "." * int((1 - rate) * 30)
            loss = train_loss.result().numpy()

            print("\r{}/{} {:^3.0f}%[{}=>{}] - loss:{:.4f}"
                  " - lbox_loss:{:.4f} - mbox_loss:{:.4f} - sbox_loss:{:.4f} - reg_loss:{:.4f}".
                  format(batch, train_steps, int(rate * 100), a, b, loss,
                         pred_loss[0], pred_loss[1], pred_loss[2], regularization_loss), end='')
            step += 1

        # 计算验证集
        for batch, (images, labels) in enumerate(valid_datasets.take(valid_steps)):
            # 得到预测，不training
            outputs = model(images)
            regularization_loss = tf.reduce_sum(model.losses)
            pred_loss = []
            for output, label, loss_fn in zip(outputs, labels, yolo_loss):
                pred_loss.append(loss_fn(label, output))

            total_valid_loss = tf.reduce_sum(pred_loss) + regularization_loss

            # 更新valid_loss
            valid_loss.update_state(total_valid_loss)

        print('\nLoss: {:.4f}, Test Loss: {:.4f}'
              ' - lbox_loss:{:.4f} - mbox_loss:{:.4f} - sbox_loss:{:.4f} - reg_loss:{:.4f}\n'.
              format(train_loss.result(), valid_loss.result(),
                     pred_loss[0], pred_loss[1], pred_loss[2], regularization_loss))

        # 保存loss，可以选择train的loss
        history_loss.append(valid_loss.result().numpy())

        # 保存到tensorboard里
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=optimizer.iterations)
            tf.summary.scalar('valid_loss', valid_loss.result(), step=optimizer.iterations)
            tf.summary.scalar('regularization_loss', regularization_loss, step=optimizer.iterations)

        # 只保存最好模型
        if valid_loss.result() < best_test_loss:
            best_test_loss = valid_loss.result()
            model.save_weights(cfg.model_path)

        # EarlyStopping
        if epoch > 1 and history_loss[epoch - 2] - history_loss[epoch - 1] > min_delta:
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            tf.print("No improvement for {} times, early stopping optimization.".format(patience))
            break


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
                          label_smooth=cfg.label_smooth,
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
        cfg.batch_size = 8
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
