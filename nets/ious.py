# -*- coding: utf-8 -*-
# @File : ious.py
# @Author: Runist
# @Time : 2020/5/21 15:44
# @Software: PyCharm
# @Brief: yolov4中CIOU的实现
import tensorflow as tf
import math


def box_iou(b1, b2):
    # 13,13,3,1,4
    # 计算左上角的坐标和右下角的坐标
    b1 = tf.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # 1,n,4
    # 计算左上角和右下角的坐标
    b2 = tf.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 计算重合面积
    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # 求出预测框左上角右下角
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真实框和预测框所有的iou
    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    # 用右下角坐标 - 左上角坐标，如果大于0就是有重叠的，如果是0就没有重叠
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / (union_area + 1e-7)

    # 计算中心的差距
    center_distance = tf.reduce_sum(tf.square(b1_xy - b2_xy), axis=-1)
    # 找到包裹两个框的最小框的左上角和右下角、计算两个框对角线的距离
    enclose_mins = tf.minimum(b1_mins, b2_mins)
    enclose_maxes = tf.maximum(b1_maxes, b2_maxes)
    enclose_wh = tf.maximum(enclose_maxes - enclose_mins, 0.0)
    # 计算对角线距离
    enclose_diagonal = tf.reduce_sum(tf.square(enclose_wh), axis=-1)
    diou = iou - 1.0 * center_distance / (enclose_diagonal + 1e-7)

    v = 4 * tf.square(tf.math.atan2(b1_wh[..., 0], b1_wh[..., 1]) -
                      tf.math.atan2(b2_wh[..., 0], b2_wh[..., 1])) / (math.pi * math.pi)

    # alpha * v是一个惩罚参数
    alpha = v / (1.0 - iou + v)
    ciou = diou - alpha * v

    ciou = tf.expand_dims(ciou, -1)
    return ciou
