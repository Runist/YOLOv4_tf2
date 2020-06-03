# -*- coding: utf-8 -*-
# @File : config.py
# @Author: Runist
# @Time : 2020/5/21 10:16
# @Software: PyCharm
# @Brief: 配置文件
import numpy as np

# 相关路径信息
annotation_path = "./config/mask_train.txt"
log_dir = r".\logs\summary"
# 预训练模型的位置
pretrain_weights_path = "D:/Python_Code/YOLOv4/config/yolov4_weight.h5"
# 模型路径
model_path = "D:/Python_Code/YOLOv4/logs/model/yolov4.h5"
best_model = "D:/Python_Code/YOLOv4/logs/model/best_model.h5"

# 获得分类名
class_names = ['with_mask', 'without_mask']

# 模型相关参数
num_bbox = 3
num_classes = len(class_names)
input_shape = (416, 416)
learning_rate = 1e-3
batch_size = 2
epochs = 50

# 余弦退火的学习率
cosine_scheduler = True
pretrain = True
fine_tune = False
train_mode = "fit"  # eager(自己撰写训练方式，偏底层的方式) fit(用.fit训练)

# iou重叠忽略阈值
ignore_thresh = 0.5
iou_threshold = 0.3
score_threshold = 0.5

# 标签处理
label_smooth = 0.05

# 数据处理
valid_rate = 0.1
shuffle_size = 1
data_pretreatment = "mosaic"  # mosaic，random(单张图片的数据增强)，normal(不增强，只进行简单填充)

# 特征层相对于输入图片缩放的倍数
strides = [32, 16, 8]
# 先验框信息
anchors = np.array([(5, 9), (9, 16), (15, 26),
                    (23, 38), (34, 53), (49, 80),
                    (80, 119), (130, 179), (200, 243)],
                   np.float32)

# 先验框对应索引
anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
