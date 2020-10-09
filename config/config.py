# -*- coding: utf-8 -*-
# @File : config.py
# @Author: Runist
# @Time : 2020/5/21 10:16
# @Software: PyCharm
# @Brief: 配置文件
import numpy as np

# 相关路径信息
annotation_path = "./config/train.txt"
log_dir = r".\logs\summary"
# 预训练模型的位置
pretrain_weights_path = "D:/Python_Code/YOLOv4/config/pretrain_tiny_model.h5"
# 模型路径
model_path = "D:/Python_Code/YOLOv4/logs/model/tiny_model.h5"
best_model = "D:/Python_Code/YOLOv4/logs/model/best_model.h5"

# 获得分类名
class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
               "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# 模型相关参数
num_classes = len(class_names)
input_shape = (416, 416, 3)
learning_rate = 1e-5
batch_size = 4
epochs = 50

# 余弦退火的学习率
cosine_scheduler = False
pretrain = False
fine_tune = False
train_mode = "fit"  # eager(自己撰写训练方式，偏底层的方式) fit(用.fit训练)
backbone = "tiny-csp-darknet"

# nms与最低分数阈值
ignore_thresh = 0.5
iou_threshold = 0.3
score_threshold = 0.52

# 标签处理
label_smooth = 0.05

# 数据处理
valid_rate = 0.1
shuffle_size = 2048
data_pretreatment = "normal"  # mosaic，random(单张图片的数据增强)，normal(不增强，只进行简单填充)

# 特征层相对于输入图片缩放的倍数
strides = [32, 16, 8]


# 先验框个数、先验框信息 和 对应索引
if backbone == 'csp-darknet':
    # 我们将先验框分成三个大小，大中小，其中每个大小内又有三种尺度的先验框
    num_bbox = 3
    anchors = np.array([(5, 9), (9, 16), (15, 26),
                        (23, 38), (34, 53), (49, 80),
                        (80, 119), (130, 179), (200, 243)],
                       np.float32)
    anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

elif backbone == "tiny-csp-darknet":
    num_bbox = 2
    anchors = np.array([(7, 12), (15, 25), (27, 43),
                        (47, 73), (88, 130), (169, 220)],
                       np.float32)
    anchor_masks = [[3, 4, 5], [0, 1, 2]]
