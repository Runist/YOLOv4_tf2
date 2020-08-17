# -*- coding: utf-8 -*-
# @File : loss.py
# @Author: Runist
# @Time : 2020/5/21 15:44
# @Software: PyCharm
# @Brief: yolov4 的 loss计算

from core.ious import box_ciou, box_iou
from nets.yolo4 import yolo4_head
import config.config as cfg
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback
from keras import backend as K


def smooth_labels(y_true, e):
    """
    u（y）表示一个关于label y，且独立于观测样本x（与x无关）的固定且已知的分布:
        q’(y|x) =（1-e) * q(y|x)+ e * u(y)

    其中，e属于[0,1]。把label y的真实分布q(y|x)与固定的分布u(y)按照1-e和e的权重混合在一起，
    构成一个新的分布。这相当于对label y中加入噪声，y值有e的概率来自于分布u(k)为方便计算，
    u(y)一般服从简单的均匀分布，则u(y)=1/K，K表示模型预测类别数目。因此，公式

        q’(y|x) = (1 - e) * q(y|x) + e/K
    :param y_true:
    :param e: [0,1]的浮点数
    :return:
    """
    k = tf.cast(tf.shape(y_true)[-1], dtype=tf.float32)
    e = tf.constant(e, dtype=tf.float32)

    return y_true * (1.0 - e) + e / k


def focal(y_true, y_pred, alpha=1, gamma=2):
    """
    何凯明提出的foacl loss有助于控制正负样本的战总loss的权重、可以按照难易程度分类样本
    pt = p if y == 1 else (1 - p)
    公式FL(pt) = -α(1 - pt)^γ * log(pt)
    :param y_true:
    :param y_pred:
    :param alpha: α 范围是0 ~ 1
    :param gamma: γ
    :return:
    """
    return alpha * tf.pow(tf.abs(y_true - y_pred), gamma)


def YoloLoss(anchors, label_smooth=cfg.label_smooth, summary_writer=None, optimizer=None):
    def compute_loss(y_true, y_pred):
        # 1. 转换 y_pred -> bbox，预测置信度，各个分类的最后一层分数， 中心点坐标+宽高
        # y_pred: (batch_size, grid, grid, anchors * (x, y, w, h, obj, ...cls))
        pred_box, grid = yolo4_head(y_pred, anchors, calc_loss=True)
        pred_conf = y_pred[..., 4:5]
        pred_class = y_pred[..., 5:]

        object_mask = y_true[..., 4:5]
        true_class = y_true[..., 5:]

        if label_smooth:
            true_class = smooth_labels(true_class, label_smooth)

        # 乘上一个比例，让小框的在total loss中有更大的占比，这个系数是个超参数，如果小物体太多，可以适当调大
        box_loss_scale = 2 - y_true[..., 2:3] * y_true[..., 3:4]

        # 找到负样本群组，第一步是创建一个数组，[]
        ignore_mask = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
        object_mask_bool = tf.cast(object_mask, tf.bool)

        # 对每一张图片计算ignore_mask
        def loop_body(b, ignore_mask):
            # object_mask_bool中，为True的值，y_true[l][b, ..., 0:4]才有效
            # 最后计算除true_box的shape[box_num, 4]
            true_box = tf.boolean_mask(y_true[b, ..., 0:4], object_mask_bool[b, ..., 0])
            # 计算预测框 和 真实框（归一化后的xywh在图中的比例）的交并比
            iou = box_iou(pred_box[b], true_box)

            # 计算每个true_box对应的预测的iou最大的box
            best_iou = tf.reduce_max(iou, axis=-1)
            # 计算出来的iou如果大于阈值则不被输入到loss计算中去，这个方法可以平衡正负样本
            ignore_mask = ignore_mask.write(b, tf.cast(best_iou < cfg.ignore_thresh, tf.float32))
            return b + 1, ignore_mask

        batch_size = tf.shape(y_pred)[0]

        # while_loop创建一个tensorflow的循环体，args:1、循环条件（b小于batch_size） 2、循环体 3、传入初始参数
        # lambda b,*args: b<m：是条件函数  b,*args是形参，b<bs是返回的结果
        _, ignore_mask = tf.while_loop(lambda b, ignore_mask: b < batch_size, loop_body, [0, ignore_mask])

        # 将每幅图的内容压缩，进行处理
        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask, -1)  # 扩展维度用来后续计算loss (b,13,13,3,1,1)

        # 计算ciou损失
        raw_true_box = y_true[..., 0:4]
        ciou = box_ciou(pred_box, raw_true_box)
        ciou_loss = object_mask * box_loss_scale * (1 - ciou)

        # 如果该位置本来有框，那么计算1与置信度的交叉熵
        # 如果该位置本来没有框，而且满足best_iou<ignore_thresh，则被认定为负样本
        # best_iou<ignore_thresh用于限制负样本数量
        object_conf = tf.nn.sigmoid_cross_entropy_with_logits(object_mask, pred_conf)
        # 计算focal loss
        conf_focal = focal(object_mask, pred_conf)
        # confidence_loss = object_mask * object_conf + (1 - object_mask) * object_conf * ignore_mask
        confidence_loss = conf_focal * (object_mask * object_conf + (1 - object_mask) * object_conf * ignore_mask)

        # 预测类别损失
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(true_class, pred_class)

        # 各个损失求平均
        location_loss = tf.reduce_sum(ciou_loss) / tf.cast(batch_size, tf.float32)
        confidence_loss = tf.reduce_sum(confidence_loss) / tf.cast(batch_size, tf.float32)
        class_loss = tf.reduce_sum(class_loss) / tf.cast(batch_size, tf.float32)

        if summary_writer:
            # 保存到tensorboard里
            with summary_writer.as_default():
                tf.summary.scalar('location_loss', location_loss, step=optimizer.iterations)
                tf.summary.scalar('confidence_loss', confidence_loss, step=optimizer.iterations)
                tf.summary.scalar('class_loss', class_loss, step=optimizer.iterations)

        return location_loss + confidence_loss + class_loss
    return compute_loss


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0,
                             min_learn_rate=0):
    """
    参数：
            global_step: 上面定义的Tcur，记录当前执行的步数。
            learning_rate_base：预先设置的学习率，当warm_up阶段学习率增加到learning_rate_base，就开始学习率下降。
            total_steps: 是总的训练的步数，等于epoch*sample_count/batch_size,(sample_count是样本总数，epoch是总的循环次数)
            warmup_learning_rate: 这是warm up阶段线性增长的初始值
            warmup_steps: warm_up总的需要持续的步数
            hold_base_rate_steps: 这是可选的参数，即当warm up阶段结束后保持学习率不变，知道hold_base_rate_steps结束后才开始学习率下降
    """
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to warmup_steps.')

    # 这里实现了余弦退火的原理，设置学习率的最小值为0，所以简化了表达式
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi *
        (global_step - warmup_steps - hold_base_rate_steps) / float(total_steps - warmup_steps - hold_base_rate_steps)))

    # 如果hold_base_rate_steps大于0，表明在warm up结束后学习率在一定步数内保持不变
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to warmup_learning_rate.')
        # 线性增长的实现
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate

        # 只有当global_step 仍然处于warm up阶段才会使用线性增长的学习率warmup_rate，否则使用余弦退火的学习率learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate, learning_rate)

    learning_rate = max(learning_rate, min_learn_rate)
    return learning_rate


class WarmUpCosineDecayScheduler(Callback):
    """
    继承Callback，实现对学习率的调度
    """
    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 min_learn_rate=0,
                 # interval_epoch代表余弦退火之间的最低点
                 interval_epoch=None,
                 verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        if interval_epoch is None:
            interval_epoch = [0.05, 0.15, 0.30, 0.50]

        # 基础的学习率
        self.learning_rate_base = learning_rate_base
        # 热调整参数
        self.warmup_learning_rate = warmup_learning_rate
        # 参数显示
        self.verbose = verbose
        # learning_rates用于记录每次更新后的学习率，方便图形化观察
        self.min_learn_rate = min_learn_rate
        self.learning_rates = []

        self.interval_epoch = interval_epoch
        # 贯穿全局的步长
        self.global_step_for_interval = global_step_init
        # 用于上升的总步长
        self.warmup_steps_for_interval = warmup_steps
        # 保持最高峰的总步长
        self.hold_steps_for_interval = hold_base_rate_steps
        # 整个训练的总步长
        self.total_steps_for_interval = total_steps

        self.interval_index = 0
        # 计算出来两个最低点的间隔
        self.interval_reset = [self.interval_epoch[0]]
        for i in range(len(self.interval_epoch)-1):
            self.interval_reset.append(self.interval_epoch[i+1]-self.interval_epoch[i])
        self.interval_reset.append(1-self.interval_epoch[-1])

    # 更新global_step，并记录当前学习率
    def on_batch_end(self, batch, logs=None):
        self.global_step += 1
        self.global_step_for_interval += 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    # 更新学习率
    def on_batch_begin(self, batch, logs=None):
        # 每到一次最低点就重新更新参数
        if self.global_step_for_interval in [0]+[int(i*self.total_steps_for_interval) for i in self.interval_epoch]:
            self.total_steps = self.total_steps_for_interval * self.interval_reset[self.interval_index]
            self.warmup_steps = self.warmup_steps_for_interval * self.interval_reset[self.interval_index]
            self.hold_base_rate_steps = self.hold_steps_for_interval * self.interval_reset[self.interval_index]
            self.global_step = 0
            self.interval_index += 1

        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps,
                                      min_learn_rate = self.min_learn_rate)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))