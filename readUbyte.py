# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 12:36:10 2018

@author: william
"""
import numpy as np
import struct
def decodeIdx3UByte(idx3_ubyte_file):
    """
    用于解析idx3_ubyte文件
    :param idx3_ubyte_file 存储idx3_ubyte文件的路径
    :return: images 数据集
    """
    with open(idx3_ubyte_file, 'rb') as file:
        data = file.read()
        # 文件头信息
        offset = 0
        fmt_header = ">iiii"
        magic_number, images_number, num_rows, num_cols = struct.unpack_from(fmt_header, data, offset)
        print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, images_number, num_rows, num_cols))
        image_size = num_rows * num_cols
        offset += struct.calcsize(fmt_header)
        fmt_image = '>' + str(image_size) + 'B'
        images = np.empty((images_number, num_rows, num_cols))
        for i in range(images_number):
            if ((i + 1) % 10000 == 0):
                print('已解析 %d' % (i + 1) + '张')
            images[i] = np.array(struct.unpack_from(fmt_image, data, offset)).reshape(num_rows, num_cols)
            offset += struct.calcsize(fmt_image)
        return images

def decodeIdx1UByte(idx1_ubyte_file):
    """
    用于解析idx1_ubyte文件
    :param idx1_ubyte_file 存储idx1_ubyte文件的路径
    :return: labels 数据集
    """
    # 读取二进制数据
    with open(idx1_ubyte_file, 'rb') as file:
        bin_data = file.read()

        # 解析文件头信息，依次为魔数和标签数
        offset = 0
        fmt_header = '>ii'
        magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
        print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

        # 解析数据集
        offset += struct.calcsize(fmt_header)
        fmt_image = '>B'
        labels = np.empty(num_images)
        for i in range(num_images):
            if (i + 1) % 10000 == 0:
                print('已解析 %d' % (i + 1) + '张')
            labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
            offset += struct.calcsize(fmt_image)
        return labels