# -*- coding:utf-8 -*-
#
#Created by lee
#
#2018-04-15

import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import yolo.config as cfg

class pascal_voc(object):
    def __init__(self, rebuild=False):
        self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')
        self.data_path = os.path.join(self.devkil_path, 'VOC2007')
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        self.flipped = cfg.FLIPPED
        self.rebuild = rebuild
        self.cursor = 0 #计数，负责遍历数据集中所有数据
        self.cursor_test = 0 #计数，负责遍历数据集中所有数据
        self.epoch = 1 #计数，负责统计训练代数
        self.gt_labels = None

    def next_batches(self, gt_labels, batch_size):
        '''
        按批次加载images和labels
        '''
        images = np.zeros((batch_size, self.image_size, self.image_size, 3)) #初始化
        labels = np.zeros((batch_size, self.cell_size, self.cell_size, self.num_class + 5)) #初始化
        count = 0
        while count < batch_size:
            imname = gt_labels[self.cursor]['imname']
            flipped = gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped)
            labels[count, :, :, :] = gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            #如果cursor大于数据集长度，则cursor置0，epoch加1，并打乱数据集
            if self.cursor >= len(gt_labels):
                np.random.shuffle(gt_labels) #对数据集随机排序
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def next_batches_test(self, gt_labels, batch_size):
        '''
        按批次加载images和labels
        '''
        images = np.zeros((batch_size, self.image_size, self.image_size, 3)) #初始化
        labels = np.zeros((batch_size, self.cell_size, self.cell_size, self.num_class + 5)) #初始化

        count = 0
        while count < batch_size:
            imname = gt_labels[self.cursor_test]['imname']
            flipped = gt_labels[self.cursor_test]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped)
            labels[count, :, :, :] = gt_labels[self.cursor_test]['label']
            count += 1
            self.cursor_test += 1
            #如果cursor大于数据集长度，则cursor置0，并打乱数据集
            if self.cursor_test >= len(gt_labels):
                np.random.shuffle(gt_labels) #对数据集随机排序
                self.cursor_test = 0
        return images, labels

    def image_read(self, imname, flipped=False):
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
        return image

    def prepare(self, model):
        gt_labels = self.load_labels(model)
        '''if self.flipped:
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] = gt_labels_cp[idx]['label'][:, ::-1, :]
                print(gt_labels_cp[idx]['label'], '   ', gt_labels_cp[idx]['label'][:, ::-1, :])
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = self.image_size - 1 - gt_labels_cp[idx]['label'][i, j, 1]
            gt_labels += gt_labels_cp'''
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels

    def load_labels(self, model):
        '''
        加载图像和bounding_boxes
        返回：图像路径和bounding_boxes
        '''
        if model == 'train':
            self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit') #文件路径
            self.data_path = os.path.join(self.devkil_path, 'VOC2007')
            txtname = os.path.join(self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        if model == 'test':
            self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit-test') #文件路径
            self.data_path = os.path.join(self.devkil_path, 'VOC2007')
            txtname = os.path.join(self.data_path, 'ImageSets', 'Main', 'test.txt')

        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]

        gt_labels = [] #定义存放数据的列表
        for index in self.image_index:
            label, num = self.load_pascal_annotation(index) #加载images和boxes
            if num == 0: #如果为0，说明xml文件中没有objects
                continue
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            gt_labels.append({'imname': imname, 'label': label, 'flipped': False})
        return gt_labels

    def load_pascal_annotation(self, index):
        '''
        从xml文件中加载class和bounding_boxes
        返回：class和bounding_boxes,以及class数量
        '''
        imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg') #图像文件名
        im = cv2.imread(imname)
        h_ratio = 1.0 * self.image_size / im.shape[0] #缩放至image_size大小
        w_ratio = 1.0 * self.image_size / im.shape[1]

        label = np.zeros((self.cell_size, self.cell_size, self.num_class + 5)) #初始化label变量
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml') #标记文件名
        tree = ET.parse(filename) #解析xml文件
        objs = tree.findall('object')

        for obj in objs:
            bbox = obj.find('bndbox')
            #读取boxes坐标和class
            x1 = max(min((float(bbox.find('xmin').text)) * w_ratio, self.image_size), 0)
            y1 = max(min((float(bbox.find('ymin').text)) * h_ratio, self.image_size), 0)
            x2 = max(min((float(bbox.find('xmax').text)) * w_ratio, self.image_size), 0)
            y2 = max(min((float(bbox.find('ymax').text)) * h_ratio, self.image_size), 0)
            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]

            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1] #boxes的中心点坐标、宽和高
            x_ind = int(boxes[0] * self.cell_size / self.image_size) #对应到cell中的第几个cell
            y_ind = int(boxes[1] * self.cell_size / self.image_size) #对应到cell中的第几个cell
            if label[y_ind, x_ind, 0] == 1: #如果cell已有数据，则继续，否则添加数据
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + cls_ind] = 1

        return label, len(objs)
