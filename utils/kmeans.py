# -*- coding: utf-8 -*-
# @File : kmeans.py
# @Author: Runist
# @Time : 2020/4/22 15:24
# @Software: PyCharm
# @Brief: K-Means计算先验框，和之前理解的不太一样，yolo的kmeans条件是需要考虑iou的


import numpy as np
import cv2 as cv
import os
import config.config as cfg
from xml.etree import ElementTree
import glob


def iou(box, clusters):
    """
    计算一个ground truth边界盒和k个先验框(Anchor)的交并比(IOU)值。
    参数box: 元组或者数据，代表ground truth的长宽。
    参数clusters: 形如(k,2)的numpy数组，其中k是聚类Anchor框的个数
    返回：ground truth和每个Anchor框的交并比。
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_


def avg_iou(boxes, clusters):
    """
    计算一个ground truth和k个Anchor的比值
    :param boxes:
    :param clusters:
    :param k:
    :return:
    """
    accuracy = np.mean([np.max(iou(box, clusters)) for box in boxes])
    return accuracy


def kmeans(boxes, k, dist=np.median):
    """
    利用IOU值进行K-means聚类
    :param boxes: 形状为(box_number, 2)的ground truth框，其中box_number是ground truth的个数
    :param k: Anchor的个数
    :param dist:
    :return: 形状为(k, 2)的k个Anchor框
    """
    # 即是上面提到的r
    box_number = boxes.shape[0]
    # 距离数组，计算每个ground truth和k个Anchor的距离
    distances = np.empty((box_number, k))
    # 上一次每个ground truth"距离"最近的Anchor索引
    last_nearest = np.zeros((box_number,))

    # 初始化聚类中心，k个簇，从r个ground truth随机选k个
    clusters = boxes[np.random.choice(box_number, k, replace=False)]  # 初始化选取类中心
    while True:

        # 计算每个ground truth和k个Anchor的距离，用1-IOU(box,anchor)来计算
        for i in range(box_number):
            distances[i] = 1 - iou(boxes[i], clusters)

        # distances = 1 - iou(boxes, clusters)
        # 对每个ground truth，选取距离最小的那个Anchor，并存下索引
        current_nearest = np.argmin(distances, axis=1)

        # 用 “==” 判断两个array 是否相同，返回的是True或False，再用.all方法判断是否全等。
        if (last_nearest == current_nearest).all():
            # 聚类中心不再更新，退出
            break

        for cluster in range(k):
            # 更新类中心
            clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)

        last_nearest = current_nearest

    return clusters


def load_dataset(path):
    """
    # 加载自己的数据集，只需要所有labelimg标注出来的xml文件即可
    :param path:
    :return:
    """
    dataset = []
    for xml_file in glob.glob("{}/*xml".format(path)):
        tree = ElementTree.parse(xml_file)
        # 图片高度
        height = int(tree.findtext("./size/height"))
        # 图片宽度
        width = int(tree.findtext("./size/width"))

        for obj in tree.iter("object"):
            # 偏移量
            xmin = float(obj.findtext("bndbox/xmin")) / width
            ymin = float(obj.findtext("bndbox/ymin")) / height
            xmax = float(obj.findtext("bndbox/xmax")) / width
            ymax = float(obj.findtext("bndbox/ymax")) / height
            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            if xmax == xmin or ymax == ymin:
                print(xml_file)
            # 将Anchor的长宽放入dateset，运行kmeans获得Anchor
            dataset.append([xmax - xmin, ymax - ymin])
    return np.array(dataset)


if __name__ == '__main__':
    CLUSTERS = 6                    # 聚类数量，anchor数量
    INPUTDIM = cfg.input_shape[0]   # 输入网络大小
    ANNOTATIONS_PATH = "D:/Python_Code/Mask_detection/MaskDetection/annotations"     # xml文件所在文件夹

    data = load_dataset(ANNOTATIONS_PATH)
    out = kmeans(data, k=CLUSTERS)
    print('Boxes:')
    print(np.array(out)*INPUTDIM)

    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    final_anchors = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Before Sort Ratios:\n {}".format(final_anchors))
    print("After Sort Ratios:\n {}".format(sorted(final_anchors)))






