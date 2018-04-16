# -*- coding:utf-8 -*-
#
#Created by lee
#
#2018-04-17

import tensorflow as tf
import datetime
import time
import os
import argparse
import yolo.config as cfg
from six.moves import xrange
from yolo.yolo_tiny_net import YOLO_tiny
from utils.pascal_voc import pascal_voc

class Solver(object):

    def __init__(self, net, data):
        self.net = net
        self.data = data
        self.batch_size = cfg.BATCH_SIZE
        self.weights_file = os.path.join(cfg.OUTPUT_DIR, cfg.WEIGHTS) #weights文件路径
        self.max_step = cfg.MAX_STEP
        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.summary_step = cfg.SUMMARY_STEP
        self.save_step = cfg.SAVE_STEP
        self.output_dir = cfg.OUTPUT_DIR
        self.save_cfg() #写配置文件

        variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(variable_to_restore)
        #self.saver = tf.train.Saver(variable_to_restore[:-3])

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        tf.summary.scalar('learning_rate', self.learning_rate)
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
        #    self.net.total_loss, global_step=self.global_step)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(
            self.net.total_loss, global_step = self.global_step) #梯度下降优化算法
        self.ema = tf.train.ExponentialMovingAverage(decay=0.999) #动量
        self.averages_op = self.ema.apply(tf.trainable_variables())
        with tf.control_dependencies([self.optimizer]):
            self.train_op = tf.group(self.averages_op)

        #配置GPU
        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        print('Restore weights from:', self.weights_file)
        self.saver.restore(self.sess, self.weights_file)

        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir)
        self.writer.add_graph(self.sess.graph)

        #self.saver = tf.train.Saver(variable_to_restore)

    def train(self):
        gt_labels = self.data.prepare('train') #读入训练用的数据
        gt_labels_t = self.data.prepare('test') #读入测试用的数据

        start_time = time.time() #开始训练时间

        for step in xrange(0, self.max_step + 1):
            images, labels = self.data.next_batches(gt_labels, self.batch_size) #读入images和labels用于训练
            feed_dict = {self.net.images: images, self.net.labels: labels}

            if step % self.summary_step == 0:
                if step % (self.summary_step * 5) == 0:
                    summary_str, loss, _ = self.sess.run([self.summary_op, self.net.total_loss, self.train_op], feed_dict=feed_dict)

                    '''
                    sum_loss为计算的损失，group为每次测试数据的组数，每组数据有batch_size个
                    即：sum_loss为 group*batch_size 张图片的损失，sum_loss/10为测试集损失
                    '''
                    sum_loss = 0
                    group = 10 
                    for num_ in xrange(0, group):
                        test_images, test_labels = self.data.next_batches_test(gt_labels_t, self.batch_size) #读入images和labels用于测试
                        feed_dict_test = {self.net.images: test_images, self.net.labels: test_labels}
                        loss_t = self.sess.run(self.net.total_loss, feed_dict=feed_dict_test)
                        sum_loss += loss_t

                    log_str = ('{} Epoch: {}, Step: {}, Loss_train: {:.4f}, Loss_test: {:.4f}, Remain: {}').format(
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.data.epoch, step, loss,
                        1. * sum_loss / group, self.remain(start_time, step))
                    print(log_str) #打印日志

                else:
                    summary_str, _ = self.sess.run([self.summary_op, self.train_op], feed_dict=feed_dict)

                self.writer.add_summary(summary_str, step) #输出到tensorboard
                if loss >= 1e4:
                    break
            else:
                self.sess.run(self.train_op, feed_dict=feed_dict)

            if step % self.save_step == 0:
                self.saver.save(self.sess, self.output_dir + '/YOLO_tiny.ckpt', global_step = step) #保存训练weights

        self.sess.close()

    def save_cfg(self):
        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)

    def remain(self, start_t, step):
        if step == 0:
            remain_time = 0
        else:
            remain_time = (time.time() - start_t) * (self.max_step - step) / step

        return str(datetime.timedelta(seconds=int(remain_time)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="yolo_tiny.ckpt", type=str)
    parser.add_argument('--gpu', default='', type=str) #指定使用GPU号
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu

    if args.weights is not None:
        cfg.WEIGHTS = args.weights

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    yolo = YOLO_tiny()
    pascal = pascal_voc()

    solver = Solver(yolo, pascal)

    print('Start training ...')
    solver.train()
    print('Done training.')

if __name__ == '__main__':
    main()
