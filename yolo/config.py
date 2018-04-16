# -*- coding:utf-8 -*-
#
#Created by lee
#
#2018-04-17

import os
#
# path and dataset parameter
#
DATA_PATH = 'data'
PASCAL_PATH = os.path.join(DATA_PATH, 'Pascal_voc')
OUTPUT_DIR = os.path.join(DATA_PATH, 'output') #输出文件路径
WEIGHTS = 'yolo_tiny.ckpt'

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

FLIPPED = True


#
# model parameter
#

IMAGE_SIZE = 448 #输入图像大小
CELL_SIZE = 7 #输出图像大小
BOXES_PER_CELL = 2 #每个cell预测box数量

ALPHA = 0.1 #leaky_ReLU系数
#DISP_CONSOLE = False

#
#损失系数
#
OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0

#
# solver parameter
#

GPU = ''

#学习速率衰减公式：learning_rate*decay_rate^(step/decay_steps)
LEARNING_RATE = 0.0002 #初始学习速率
DECAY_STEPS = 20000
DECAY_RATE = 0.1 #学习速率衰减系数
STAIRCASE = True

BATCH_SIZE = 32
MAX_STEP = 30000 #最大步长
SUMMARY_STEP = 10 #每隔 summary_step 步输出到TensorBoard，每隔 summary_step*5 步打印输出
SAVE_STEP = 50 #每隔save_step步保存训练weights

#
# test parameter
#

THRESHOLD = 0.2 #阈值
IOU_THRESHOLD = 0.5 #IOU阈值
