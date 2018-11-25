#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image, ImageDraw


def print_start():
    """
    打印 开始 logog
    :return:
    """

    print("===================================================================")
    print(r"""

              _                            _                 
             | |                          | |                
__      _____| | ___ ___  _ __ ___   ___  | |__  _   _  __ _ 
\ \ /\ / / _ \ |/ __/ _ \| '_ ` _ \ / _ \ | '_ \| | | |/ _` |
 \ V  V /  __/ | (_| (_) | | | | | |  __/ | | | | |_| | (_| |
  \_/\_/ \___|_|\___\___/|_| |_| |_|\___| |_| |_|\__,_|\__,_|

    """)
    print("====================================================================")

def AddSmudginess(img, Smu):
    rows = r(Smu.shape[0] - 50)

    cols = r(Smu.shape[1] - 50)
    adder = Smu[rows:rows + 50, cols:cols + 50]
    adder = cv2.resize(adder, (50, 50))
    #   adder = cv2.bitwise_not(adder)
    img = cv2.resize(img, (50, 50))
    img = cv2.bitwise_not(img)
    img = cv2.bitwise_and(adder, img)
    img = cv2.bitwise_not(img)
    return img


def rot(img, angel, shape, max_angel):
    """ 使图像轻微的畸变
        img 输入图像
        factor 畸变的参数
        size 为图片的目标尺寸
    """
    size_o = [shape[1], shape[0]]
    size = (shape[1] + int(shape[0] * np.cos((float(max_angel) / 180) * 3.14)), shape[0])
    interval = abs(int(np.sin((float(angel) / 180) * 3.14) * shape[0]))
    pts1 = np.float32([[0, 0], [0, size_o[1]], [size_o[0], 0], [size_o[0], size_o[1]]])
    if (angel > 0):
        pts2 = np.float32([[interval, 0], [0, size[1]], [size[0], 0], [size[0] - interval, size_o[1]]])
    else:
        pts2 = np.float32([[0, 0], [interval, size[1]], [size[0] - interval, 0], [size[0], size_o[1]]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)

    return dst


def rotRandrom(img, factor, size):
    """
    根据输入和输出点获得图像透视变换的矩阵
    :param img:
    :param factor:
    :param size:
    :return:
    """
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)], [r(factor), shape[0] - r(factor)], [shape[1] - r(factor), r(factor)],
                       [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)
    return dst


def tfactor(img):
    """
    颜色空间转换

    :param img:
    :return:
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    hsv[:, :, 0] = hsv[:, :, 0] * (0.8 + np.random.random() * 0.2)
    hsv[:, :, 1] = hsv[:, :, 1] * (0.3 + np.random.random() * 0.7)
    hsv[:, :, 2] = hsv[:, :, 2] * (0.2 + np.random.random() * 0.8)

    img = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    return img


def random_envirment(img, data_set):
    index = r(len(data_set))
    env = cv2.imread(data_set[index])
    env = cv2.resize(env, (img.shape[1], img.shape[0]))
    bak = (img == 0)
    bak = bak.astype(np.uint8) * 255
    inv = cv2.bitwise_and(bak, env)
    img = cv2.bitwise_or(inv, img)
    return img


def GenCh(f, val):
    img = Image.new("RGB", (45, 70), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 3), val, (0, 0, 0), font=f)
    img = img.resize((23, 70))
    A = np.array(img)

    return A


def GenCh1(f, val):
    img = Image.new("RGB", (23, 70), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 2), val, (0, 0, 0), font=f)
    A = np.array(img)
    return A


def AddGauss(img, level):
    """
    均值滤波，模拟模糊图片
    :param img:
    :param level:
    :return:
    """
    return cv2.blur(img, (level * 2 + 1, level * 2 + 1))


def r(val):
    return int(np.random.random() * val)


def AddNoiseSingleChannel(single, r_value):
    diff = 255 - single.max()
    noise = np.random.normal(0, 1 + r(r_value), single.shape)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = diff * noise
    noise = noise.astype(np.uint8)
    dst = single + noise
    return dst


def addNoise(img, sdev=0.5, avg=10, r_value=6):
    """
    增加噪声
    :param img:
    :param sdev:
    :param avg:
    :return:
    """
    img[:, :] = AddNoiseSingleChannel(img[:, :], r_value)
    return img


def add_two_img(big, smaller, rgb=(0, 0, 0)):
    """
    将 两个图片融合

    :param self:
    :param big:
    :param smaller:
    :param rgb:
    :return:
    """

    big_x, big_y, _ = big.shape
    sim_x, sim_y, _ = smaller.shape
    x_offset, y_offset = int((big_x - sim_x) / 2), int((big_y - sim_y) / 2)
    p = big[x_offset:sim_x + x_offset, y_offset:sim_y + y_offset, :]
    p[:][:] = rgb
    big[x_offset:sim_x + x_offset, y_offset:sim_y + y_offset, :] = smaller + p
    return big


def gen_rectangle_and_ring(img, pt1, pt2, color, r):
    """
    生成带圆角的矩形框
    :param img:
    :param pt1:
    :param pt2:
    :param color:
    :param r:
    :return:
    """
    img = cv2.rectangle(img, pt1=(pt1[0] + r, pt1[1]), pt2=(pt2[0] - r, pt2[1]), color=color)
    img = cv2.circle(img, center=(r, r), radius=r, color=color)
    img = cv2.circle(img, center=(pt2[0] - r, r), radius=r, color=color)
    return img


def random_add_with_alpha(img, data_set):
    """

    :param img:
    :param data_set:
    :return:
    """
    index = r(len(data_set))
    env = cv2.imread(data_set[index], cv2.IMREAD_GRAYSCALE)
    env = cv2.resize(env, (img.shape[1], img.shape[0]))
    return add_with_alpha(img, env)


def add_with_alpha(bottom, top, a=0.6, b=0.4):
    # 权重越大，透明度越低
    overlapping = cv2.addWeighted(bottom, a, top, b, 0)
    return overlapping


if __name__ == '__main__':
    img = cv2.imread("/Volumes/doc/projects/ml/ocr_tensorflow_cnn/lib/vin_generator/images/back_2.jpeg")
    height, weight, _ = img.shape
    r = int(height / 2)
    cv2.imshow("test", gen_rectangle_and_ring(img, (0, 0), (weight, height), (0, 0, 0), r))
    cv2.waitKeyEx(0)
