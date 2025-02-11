#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 下午4:58
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_data_processor.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet的数据解析类
"""
import tensorflow as tf
import os

from ..config import global_config

CFG = global_config.cfg
VGG_MEAN = [123.68, 116.779, 103.939]


class DataSet(object):
    """
    实现数据集类
    """

    def __init__(self, dataset_info_file, data_bp=""):
        """
        :param dataset_info_file:
        """
        self._len = 0
        self.dataset_info_file = dataset_info_file
        self._data_bp = data_bp
        self._img, self._label_instance, self._label_existence = self._init_dataset()

    def __len__(self):
        return self._len

    @staticmethod
    def process_img(img_queue):
#        try: 
        img_raw = tf.read_file(img_queue)
        img_decoded = tf.image.decode_jpeg(img_raw, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH],
                                             method=tf.image.ResizeMethod.BICUBIC)
        img_casted = tf.cast(img_resized, tf.float32)
        return tf.subtract(img_casted, VGG_MEAN)
#        except tf.errors.NotFoundError: 
#            print("img_queue", img_queue)
#            #raise FileNotFoundError("Could not find image with uri %s" % img_queue)

    @staticmethod
    def process_label_instance(label_instance_queue):
        label_instance_raw = tf.read_file(label_instance_queue)
        label_instance_decoded = tf.image.decode_png(label_instance_raw, channels=1)
        label_instance_resized = tf.image.resize_images(label_instance_decoded,
                                                        [CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH],
                                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        label_instance_resized = tf.reshape(label_instance_resized, [CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH])
        return tf.cast(label_instance_resized, tf.int32)

    @staticmethod
    def process_label_existence(label_existence_queue):
        return tf.cast(label_existence_queue, tf.float32)

    def _init_dataset(self):
        """
        :return:
        """
        if not tf.gfile.Exists(self.dataset_info_file):
            raise ValueError('Failed to find file: ' + self.dataset_info_file)

        img_list = []
        label_instance_list = []
        label_existence_list = []

        with open(self.dataset_info_file, 'r') as file:
            for _info in file:
                info_tmp = _info.strip(' ').split()
                imfile, laneseg_file, l1_exist, l2_exist, l3_exist, l4_exist = info_tmp
                #img_list.append(info_tmp[0][1:])
                img_list.append(os.path.join(self._data_bp, imfile))
                label_instance_list.append(os.path.join(self._data_bp, laneseg_file))
                label_existence_list.append([int(l1_exist), int(l2_exist), int(l3_exist), int(l4_exist)])

        self._len = len(img_list)
        # img_queue = tf.train.string_input_producer(img_list)
        # label_instance_queue = tf.train.string_input_producer(label_instance_list)
        with tf.name_scope('data_augmentation'):
            image_tensor = tf.convert_to_tensor(img_list)
            label_instance_tensor = tf.convert_to_tensor(label_instance_list)
            label_existence_tensor = tf.convert_to_tensor(label_existence_list)
            input_queue = tf.train.slice_input_producer([image_tensor, label_instance_tensor, label_existence_tensor])
            img = self.process_img(input_queue[0])
            label_instance = self.process_label_instance(input_queue[1])
            label_existence = self.process_label_existence(input_queue[2])

        return img, label_instance, label_existence

    def next_batch(self, batch_size):
        return tf.train.batch([self._img, self._label_instance, self._label_existence], batch_size=batch_size,
                              num_threads=CFG.TRAIN.CPU_NUM)
