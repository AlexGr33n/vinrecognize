#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 @author hsjfans 
"""
import tensorflow as tf
from lib.gen.generator import GenVin
import numpy as np

class NetWork():

    def __init__(self, width, height, label_size, classify_size, gen: GenVin,checkpoint='./checkpoint/'):
        """

        :param width: 输入的图片宽度
        :param height: 输入的图片高度
        :param label_size: 输入的标签长度
        :param classify_size: 分类的种类
        :param checkpoint: checkout point path
        """
        self.width = width
        self.height = height
        self.label_size = label_size
        self.classify_size = classify_size
        self.input = tf.placeholder(tf.float32, [None, self.height * self.width])
        self.labels = tf.placeholder(tf.float32, [None, self.classify_size * self.label_size])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout
        # 是否在训练阶段
        self.train_phase = tf.placeholder(tf.bool)
        self.checkpoint = checkpoint
        self.gen = gen

    # See __build_net_work()
    def build(self, w_alpha=0.01, b_alpha=0.1):
        self.__build_net_work(w_alpha, b_alpha)

    def start(self, train=True,w_alpha=0.01, b_alpha=0.1):
        """
        开始训练
        :param train:
        :return:
        """
        self.__build_net_work(w_alpha=w_alpha, b_alpha=b_alpha)
        if train:
            self.__train()
        else:
            self.__predict()

    def __build_net_work(self, w_alpha, b_alpha):
        """
        建立 网络结构
        :param w_alpha: w 学习率
        :param b_alpha: b 学习率
        :return:
        """
        # 输入
        x = tf.reshape(self.input, shape=[-1, self.height, self.width, 1])
        w_c1 = tf.Variable(w_alpha * tf.random_normal([5, 5, 1, 32]))
        b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
        conv1 = tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1)
        conv1 = self.batch_norm(conv1, tf.constant(0.0, shape=[32]),
                                tf.random_normal(shape=[32], mean=1.0, stddev=0.02),
                                self.train_phase, scope='bn_1')
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, self.keep_prob)
        print(conv1.shape)
        w_c2 = tf.Variable(w_alpha * tf.random_normal([5, 5, 32, 64]))
        b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2)
        conv2 = self.batch_norm(conv2, tf.constant(0.0, shape=[64]),
                                tf.random_normal(shape=[64], mean=1.0, stddev=0.02),
                                self.train_phase, scope='bn_2')
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_prob)
        print(conv2.shape)
        w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
        b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv3 = tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3)
        conv3 = self.batch_norm(conv3, tf.constant(0.0, shape=[64]),
                                tf.random_normal(shape=[64], mean=1.0, stddev=0.02),
                                self.train_phase, scope='bn_3')
        conv3 = tf.nn.relu(conv3)
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_prob)
        print(conv3.shape)
        w_c4 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
        b_c4 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv4 = tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4)
        conv4 = self.batch_norm(conv4, tf.constant(0.0, shape=[64]),
                                tf.random_normal(shape=[64], mean=1.0, stddev=0.02),
                                self.train_phase, scope='bn_4')
        conv4 = tf.nn.relu(conv4)
        conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv4 = tf.nn.dropout(conv4, self.keep_prob)
        print(conv4.shape)
        # Fully connected layer
        w_d = tf.Variable(w_alpha * tf.random_normal([4 * 17 * 64, 1024]))
        b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv4, [-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, self.keep_prob)

        w_out = tf.Variable(w_alpha * tf.random_normal([1024, self.label_size * self.classify_size]))
        b_out = tf.Variable(b_alpha * tf.random_normal([self.label_size * self.classify_size]))
        self.output = tf.add(tf.matmul(dense, w_out), b_out)

    def __train(self, learning_rate=0.002, close_acc=0.95, log_size=100, saver_size=1000):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.labels))
        # 最后一层用来分类的softmax和sigmoid有什么不同？
        # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        predict = tf.reshape(self.output, [-1, self.label_size, self.classify_size])
        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(tf.reshape(self.labels, [-1, self.label_size, self.classify_size]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # 初始化所有 权重
            sess.run(tf.global_variables_initializer())
            step = 0
            while True:
                batch_x, batch_y, _ = self.gen.get_next_batch()
                _, loss_ = sess.run([optimizer, loss],
                                    feed_dict={self.input: batch_x, self.labels: batch_y, self.keep_prob: 0.75,
                                               self.train_phase: True})
                print("第%s步，loss 为：%s" % (step, loss_))
                # 每100 step计算一次准确率
                if step % log_size == 0 and step != 0:
                    batch_x_test, batch_y_test, text = self.gen.get_next_batch()
                    acc, predict = sess.run([accuracy, max_idx_p],
                                            feed_dict={self.input: batch_x_test, self.labels: batch_y_test,
                                                       self.keep_prob: 1.,
                                                       self.train_phase: False})
                    self.log_test(text, predict)
                    print("第%s步，训练准确率为：%s" % (step, acc))
                    # 每 1000 step 保存一次模型
                    if step % saver_size == 0:
                        if acc > close_acc:
                            saver.save(sess, self.checkpoint, global_step=step)
                            break
                        else:
                            saver.save(sess, self.checkpoint, global_step=step)
                step += 1

    def __predict(self):
        input, text = self.gen.get_predict_source()
        output = self.output()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.__restore(saver, sess)
            predict = tf.reshape(output, [-1, self.label_size, self.classify_size])
            max_idx_p = tf.argmax(predict, 2)
            text_list = sess.run(max_idx_p,
                                 feed_dict={self.input: input, self.keep_prob: 1, self.train_phase: False})
            self.log_test(text, text_list)

    # Batch Normalization? 有空再理解,tflearn or slim都有封装
    ## http://stackoverflow.com/a/34634291/2267819
    def batch_norm(self, x, beta, gamma, phase_train, scope='bn', decay=0.9, eps=1e-5):
        """
        防止 OverFitting
        :param x: input
        :param beta:
        :param gamma:
        :param phase_train:
        :param scope:
        :param decay:
        :param eps:
        :return:
        """
        with tf.variable_scope(scope):
            # beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0), trainable=True)
            # gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, stddev), trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=decay)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train, mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
        return normed

    def log_test(self, labels, predicts):
        """
        打印
        :param labels: 实际标签
        :param predicts:  输出标签
        :return:
        """
        for idx, item in enumerate(predicts):
            text = item.tolist()
            vector = np.zeros(self.label_size * self.classify_size)
            i = 0
            for n in text:
                vector[i * self.classify_size + n] = 1
                i += 1
            predict_text = self.gen.vec2text(vector)
            print("正确: {}  预测: {}".format(''.join(labels[idx]), predict_text))


    def __restore(self, saver, sess):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            raise Exception("loading error")
