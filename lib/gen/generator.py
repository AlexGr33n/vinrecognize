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

# 路径先采用绝对路径
test_gen_path = '/Volumes/doc/projects/ml/vin_recognize/static/test/'
images_path = "/Volumes/doc/projects/ml/vin_recognize/static/images/"
font_path = '/Volumes/doc/projects/ml/vin_recognize/static/font/'
bad_path = '/Volumes/doc/projects/ml/vin_recognize/static/bad/'
predict_path = '/Volumes/doc/projects/ml/vin_recognize/static/predict/'
font = 'OCR-B.ttf'


class GenVin:

    def __init__(self, height, width, predict_path=predict_path, fontSize=30,
                 NoPlates=bad_path, font_path=font_path,
                 images_path=images_path, font=font):

        self.fontSize = fontSize
        self.font = ImageFont.truetype(font_path + font, self.fontSize, 0)
        self.height = height
        self.width = width
        # 字体 背景
        # self.img = np.array(Image.new("RGB", (442, 98), (0, 0, 0)))
        # 作为背景使用
        self.bg = cv2.imread(images_path + "back.png", cv2.IMREAD_GRAYSCALE)
        # 环境使用
        # self.env = cv2.imread(images_path + "env.png",cv2.IMREAD_ANYCOLOR)
        # print(self.env.shape)
        # 512 98
        # self.img = add_two_img(self.bg,self.env)
        self.predict_path = predict_path
        self.len = len(chars)
        self.max_size = 17  # 字体长度
        # # 增加干扰，灯光等
        self.bad_paths = []
        if NoPlates is not None:
            for parent, parent_folder, filenames in os.walk(NoPlates):
                for filename in filenames:
                    path = parent + "/" + filename
                    self.bad_paths.append(path)
        # self.bg = random_envirment(self.bg, self.bad_paths)

    def draw(self, val):
        """
        画出 所有字符
        :param val:
        :return:
        """
        height, width = self.bg.shape
        offset = 0
        base = int((width - self.max_size * self.fontSize) / 2)
        for i, item in enumerate(val):
            self.bg[int((height - self.fontSize) / 2):int((height + self.fontSize) / 2),
            base: base + self.fontSize] = self.draw_item(item, 0, 0)
            base += self.fontSize + offset
        return self.bg

    def draw_item(self, val, left_offset, top_offset):
        """
                生成一个字符
                :param val:
                :return:
                """
        img = Image.new("L", (self.fontSize, self.fontSize), (240))
        draw = ImageDraw.Draw(img)
        draw.text((left_offset, top_offset), val, (0), font=self.font)
        # img = img.resize((24, 102))
        A = np.array(img)
        # img.save("213.png")
        return A

    def GenCh(self, val, img):
        """
        生成一个字符
        :param val:
        :return:
        """
        img = Image.new("RGB", (self.fontSize, self.bg.shape[1]), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((0, int(self.bg.shape[1] - self.fontSize) / 2), val, (255, 255), font=self.font)
        A = np.array(img)
        return A

    def random_start(self, seed):
        ran = r(seed)
        return ran % 3 == 0

    def generate(self, text):
        fg = self.draw(text)
        if self.random_start(3):
            fg = cv2.bitwise_not(fg)
        # # 背景图
        com = fg
        # com = add_two_img(self.bg,self.env)
        # print(com.shape)
        if self.random_start(3):
            com = rot(com, r(30) - 10, com.shape, 10)
        # com = AddSmudginess(com,self.smu)
        # if self.random_start(3):
        com = tfactor(com)
        # com = random_add_with_alpha(com, self.bad_paths)
        if self.random_start(3):
            com = rotRandrom(com, 2, (com.shape[1], com.shape[0]))
        # # 增加噪点
        if self.random_start(5):
            com = AddGauss(com, 1 + r(1))
        if self.random_start(5):
            com = addNoise(com)
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
        return img[:, :], text, label  # 返回的label为标签，img为深度为3的图像像素

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
            filename = os.path.join(outputPath, str(i).zfill(4) + '.' + "".join(plateStr) + ".png")
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


generator = GenVin(49, 258)
if __name__ == '__main__':
    # generator = GenVin("./font/DejaVuSansMono.ttf",80,440)
    # print("".join(generator.get_text()))
    # generator = GenVin("./font/Bitter-Regular.ttf")
    # generator = GenVin("./font/SimHei.ttf")
    generator.gen_Batch(3, test_gen_path)