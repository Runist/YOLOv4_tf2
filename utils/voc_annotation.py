# -*- coding: utf-8 -*-
# @File : voc_annotation.py
# @Author: Runist
# @Time : 2020/5/8 10:48
# @Software: PyCharm
# @Brief: voc转换为yolo4读取的格式


import xml.etree.ElementTree as ET
import config.config as cfg
import random
import os
import re


def convert_annotation(xml_path, image):
    """
    把单个xml转换成annotation格式
    :param xml_path: 标签文件路径
    :param image: 图片id
    :return: bbox: 先验框的坐标信息
    """
    image_id = re.findall(r'(.+?)\.', image)[0]
    in_file = open('{}/{}.xml'.format(xml_path, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()

    bbox = ''

    for obj in root.iter('object'):

        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # 不在分类内部的不要，难度为1的也不要
        if cls not in cfg.class_names or int(difficult) == 1:
            continue

        cls_id = cfg.class_names.index(cls)
        xmlbox = obj.find('bndbox')

        b = (int(xmlbox.find('xmin').text),
             int(xmlbox.find('ymin').text),
             int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))

        bbox += " " + ",".join([str(a) for a in b]) + ',' + str(cls_id)

    return bbox


if __name__ == '__main__':
    # VOC数据集的路径
    xml_path = 'D:/Python_Code/Dataset/VOCdevkit/VOC2012/Annotations'
    image_path = 'D:/Python_Code/Dataset/VOCdevkit/VOC2012/JPEGImages'

    total_xml = os.listdir(xml_path)
    total_img = os.listdir(image_path)

    train_percent = 0.9
    test_percent = 1 - train_percent

    # 生成一段序列，然后在序列中随机选索引
    num = len(total_xml)
    image_range = range(num)
    tr = int(num * train_percent)
    te = int(num * test_percent)

    # 第一个参数序列，第二个参数去除的个数，从a中取出n个数字
    train_num = random.sample(image_range, tr)

    train_ids = []
    test_ids = []

    # 训练集和测试集分开
    for i in image_range:
        name = total_img[i]

        if i in train_num:
            train_ids.append(name)
        else:
            test_ids.append(name)

    # 将信息写入train.txt和test.txt
    image_ids = {"train": train_ids, "test": test_ids}
    for key, value in image_ids.items():
        files = open('../config/{}.txt'.format(key), 'w')
        for image in value:
            print(image)
            bbox = convert_annotation(xml_path, image)
            if len(bbox) == 0:
                continue

            # 训练图片实际路径
            files.write('{}/{}'.format(image_path, image))
            files.write(bbox)

            files.write('\n')
        files.close()
