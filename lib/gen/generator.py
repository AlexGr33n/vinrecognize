#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
注意，训练使用，不做具体限制

生成 训练的图片

"""
import os

from PIL import ImageFont
from lib.common import *
from lib.gen.static import *

# 路径先采用绝对路径'=
back_path = '/Volumes/doc/projects/ml/vin_recognize/static/back/'
test_gen_path = '/Volumes/doc/projects/ml/vin_recognize/static/test/'
images_path = "/Volumes/doc/projects/ml/vin_recognize/static/images/"
font_path = '/Volumes/doc/projects/ml/vin_recognize/static/font/'
bad_path = '/Volumes/doc/projects/ml/vin_recognize/static/bad/'
predict_path = '/Volumes/doc/projects/ml/vin_recognize/static/predict/'
font = 'OCR-B.ttf'


class GenVin:

    # See https://stackoverflow.com/questions/2726171/how-to-change-font-size-using-the-python-imagedraw-library
    def __init__(self, height, width, predict_path=predict_path,
                 NoPlates=bad_path, font_path=font_path,
                 font=font,back_path=back_path):

        self.font_name = font
        self.font_path = font_path
        self.height = height
        self.width = width
        # print(self.img.format, self.img.size, self.img.mode)
        self.predict_path = predict_path
        self.len = len(chars)
        self.max_size = 17  # 字体长度
        # # 增加干扰，灯光等
        self.bad_paths = []
        self.back_paths = []
        if NoPlates is not None:
            for parent, parent_folder, filenames in os.walk(NoPlates):
                for filename in filenames:
                    path = parent + "/" + filename
                    self.bad_paths.append(path)
        if back_path is not None:
            for parent, parent_folder, filenames in os.walk(back_path):
                for filename in filenames:
                    path = parent + "/" + filename
                    self.back_paths.append(path)

    def random_get_back(self):
        index = r(len(self.back_paths))
        return Image.open(self.back_paths[index])  # 打开文件

    def draw(self, val, rate=1.0):
        """
        画出 所有字符
        :param val:
        :return:
        """
        self.img = self.random_get_back()
        width, height = self.img.size
        base_width = r(int(0.1 * width))
        offset = 0
        self.fontSize = int((width - 2 * base_width) / self.max_size / rate)
        self.font = ImageFont.truetype(self.font_path + self.font_name, self.fontSize)
        base = int((width - self.max_size * self.fontSize * rate + (self.max_size - 1) * offset) / 2)
        base_height = int((height - self.fontSize) / 2)
        img = self.img.copy()
        draw_model = ImageDraw.Draw(img)  # 修改图片
        distance = 0
        base_rate = 3
        p = r(10) / 10
        if r(2) == 1:
            base_rate *= -p
        else:
            base_rate *= p
        # print("base_rate is ", base_rate, np.sin(base_rate * np.pi / 180))
        for i, item in enumerate(val):
            # self.bg[int((height - self.fontSize) / 2):int((height + self.fontSize) / 2),
            # base: base + self.fontSize] = self.draw_item(item, 0, 0)
            self.dir_draw_item(draw_model, item, base, (base_height + self.bias_char_offset(distance, base_rate)))
            base += self.fontSize * rate + offset
            distance += self.fontSize * rate + offset
        A = np.array(img)
        return A

    def bias_char_offset(self, distance, base_rate):
        """
        生成倾斜 的文字
        :param base_rate: 倾斜的角度 绝对值 10
        """
        return int(distance * np.sin(base_rate * np.pi / 180))

    def dir_draw_item(self, draw, val, left_offset, top_offset):
        """

        :param val:
        :param left_offset:
        :param top_offset:
        :return:
        """
        draw.text((left_offset, top_offset), val, (255, 255, 255), font=self.font)

    def random_start(self, seed):
        ran = r(seed)
        return ran % 3 == 0

    def generate(self, text):
        # self.bg = random_envirment(self.bg,self.bad_paths)
        fg = self.draw(text)
        fg = cv2.resize(fg, (self.width, self.height))
        if self.random_start(6):
            fg = cv2.bitwise_not(fg)
        # 背景图
        com = fg
        # com = add_two_img(self.bg,self.env)
        # print(com.shape)

        # com = rot(com, r(30) - 10, com.shape, 10)
        # com = AddSmudginess(com,self.smu)
        # if self.random_start(3):
        # com = tfactor(com)
        # com = random_add_with_alpha(com, self.bad_paths)
        # com = rotRandrom(com, 2, (com.shape[1], com.shape[0]))
        # # 增加噪点
        if self.random_start(10):
            com = AddGauss(com, 1 + r(1))
        # if self.random_start(5):
        #     com = addNoise(com)
        return com

    def gen_sample(self):
        text, label = self.get_text()
        img = self.generate(text)
        img = cv2.resize(img, (self.width, self.height))
        img = np.multiply(img, 1 / 255.0)  # [height,width,channel]
        return label, img[:, :, 2]  # 返回的label为标签，img为深度为3的图像像素

    def gen_img(self):
        text, label = self.get_text()
        img = self.generate(text)
        img = cv2.resize(img, (self.width, self.height))
        img = np.multiply(img, 1 / 255.0)  # [height,width,channel]
        return img[:, :, :2], text, label  # 返回的label为标签，img为深度为3的图像像素

    # 生成一个训练batch
    def get_next_batch(self, batch_size=128):
        batch_x = np.zeros([batch_size, self.width * self.height])
        batch_y = np.zeros([batch_size, self.len * self.max_size])
        labels = []
        for i in range(batch_size):
            image, text, vec = self.gen_img()
            labels.append(text)
            batch_x[i, :] = image.reshape((self.width * self.height))
            batch_y[i, :] = vec
        return batch_x, batch_y, labels

    def get_predict_source(self):
        items = os.listdir(self.predict_path)
        batch = len(items)
        batch_x = np.zeros([batch, self.width * self.height])
        textes = []
        for i, item in enumerate(items):
            textes.append(item.split("/")[-1])
            img = cv2.imread(os.path.join(self.predict_path, item), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.width, self.height))
            batch_x[i, :] = img.reshape((self.width * self.height))

        return batch_x, textes

    def text2vec(self, text):
        text_len = len(text)
        if text_len > self.max_size:
            raise ValueError('字符最长 ', self.max_size, '个字符')
        vector = np.zeros(self.max_size * self.len)
        for i, c in enumerate(text):
            idx = i * self.len + chars_indexes[c]
            vector[idx] = 1
        return vector

    # 向量转回文本
    def vec2text(self, vec):
        char_pos = vec.nonzero()[0]
        text = []
        for i, c in enumerate(char_pos):
            char_idx = c % self.len
            text.append(chars[char_idx])
        return "".join(text)

    def gen_Batch(self, batchSize, outputPath):
        if (not os.path.exists(outputPath)):
            os.mkdir(outputPath)

        for i in range(batchSize):
            # plateStr = self.genPlateString(-1, -1)
            plateStr, label = self.get_text()
            img = self.generate(plateStr)
            img = cv2.resize(img, (self.width, self.height))
            filename = os.path.join(outputPath, str(i).zfill(4) + '.' + "".join(plateStr) + ".jpg")
            cv2.imwrite(filename, img)

    def get_text(self):
        """
         # 随机生成字串，长度固定
        # 返回text,及对应的向量
        :return:
        """
        text = []
        # vecs = np.zeros((self.max_size * self.len))
        text += (self.get_random_char(STATE_CODE, 1))  # 生成车牌
        text += (self.get_random_char(chars, 2))
        text += (self.get_random_char(FOUR_CODE, 1))  # 4
        text += (self.get_random_char(chars, 4))  # 8
        text += (self.get_random_char(NINE_CODE, 1))  # 9
        text += (self.get_random_char(chars, 8))
        return text, self.text2vec(text)

    def get_random_char(self, chars, size):
        length = len(chars)
        ints = np.random.randint(0, length, size, dtype=np.int)
        result = []
        for i in ints: result.append(chars[i])
        return result


generator = GenVin(50, 300, font='font.ttf')
if __name__ == '__main__':
    # generator = GenVin("./font/DejaVuSansMono.ttf",80,440)
    # print("".join(generator.get_text()))
    # generator = GenVin("./font/Bitter-Regular.ttf")
    # generator = GenVin("./font/SimHei.ttf")
    generator.gen_Batch(5, test_gen_path)
