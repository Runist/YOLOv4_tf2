# -*- coding: utf-8 -*-
# @File : config.py
# @Author: Runist
# @Time : 2020/5/21 10:16
# @Software: PyCharm
# @Brief: 配置文件
import numpy as np

# 相关路径信息
annotation_path = "./config/2012_train.txt"
log_dir = r".\logs\summary"
# 预训练模型的位置
pretrain_weights_path = "D:/Python_Code/YOLOv4/config/yolov4_weight.h5"
# 模型路径
model_path = "D:/Python_Code/YOLOv4/logs/model/yolov4.h5"
best_model = "D:/Python_Code/YOLOv4/logs/model/best_model.h5"

# 模型相关参数
num_bbox = 3
num_classes = 20
input_shape = (416, 416)
learning_rate = 1e-3
batch_size = 4
epochs = 50

# 余弦退火的学习率
cosine_scheduler = True
pretrain = True
fine_tune = False
train_mode = "fit"

# iou重叠忽略阈值
ignore_thresh = 0.5
iou_threshold = 0.3
score_threshold = 0.5

# 标签处理
label_smoothing = 0.05

# 数据处理
valid_rate = 0.1
shuffle_size = 1
data_pretreatment = "mosaic"


# 先验框信息
anchors = np.array([(10, 13), (16, 30), (33, 23),
                    (30, 61), (62, 45), (59, 119),
                    (116, 90), (156, 198), (373, 326)],
                   np.float32)

# 获得分类名
class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# 先验框对应索引
anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
