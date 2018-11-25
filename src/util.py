import cv2
import numpy as np
import tensorflow as tf

from lib.gen.static import chars
from lib.gen.generator import generator

# 分类类别为
num_classes = len(chars) + 1  # len(chars)  + ctc blank

# ===========  tensorflow 常量 =========== #

"""

Notice : 训练车牌和训练 vin 码，采用同一个 util 文件

使用前要进行修改

"""

tf.flags.DEFINE_boolean('restore', False, 'whether to restore from the latest checkpoint')
tf.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.flags.DEFINE_float('initial_learning_rate', 1e-3, 'initial lr')

tf.flags.DEFINE_integer('image_height', 80, 'image height')
tf.flags.DEFINE_integer('image_width', 440, 'image width')
tf.flags.DEFINE_integer('image_channel', 1, 'image channels as input')

tf.flags.DEFINE_integer('cnn_count', 4, 'count of cnn module to extract image features.')
tf.flags.DEFINE_integer('out_channels', 64, 'output channels of last layer in CNN')
tf.flags.DEFINE_integer('num_hidden', 128, 'number of hidden units in lstm')
tf.flags.DEFINE_float('output_keep_prob', 0.8, 'output_keep_prob in lstm')
tf.flags.DEFINE_integer('num_epochs', 10000, 'maximum epochs')
tf.flags.DEFINE_integer('batch_size', 128, 'the batch_size')
tf.flags.DEFINE_integer('batch', 10, 'the batch_size')
tf.flags.DEFINE_integer('save_steps', 1000, 'the step to save checkpoint')
tf.flags.DEFINE_float('leakiness', 0.01, 'leakiness of lrelu')
tf.flags.DEFINE_integer('validation_steps', 500, 'the step to validation')

tf.flags.DEFINE_float('decay_rate', 0.98, 'the lr decay rate')
tf.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')

tf.flags.DEFINE_integer('decay_steps', 10000, 'the lr decay_step for optimizer')
tf.flags.DEFINE_float('momentum', 0.9, 'the momentum')

tf.flags.DEFINE_string('train_dir', './imgs/train/', 'the train data dir')
tf.flags.DEFINE_string('val_dir', './imgs/val/', 'the val data dir')
tf.flags.DEFINE_string('infer_dir', './imgs/infer/', 'the infer data dir')
tf.flags.DEFINE_string('log_dir', './log', 'the logging dir')
tf.flags.DEFINE_string('mode', 'train', 'train, val or infer')
tf.flags.DEFINE_integer('num_gpus', 0, 'num of gpus')
FLAGS = tf.flags.FLAGS

next_train_img_path = "/Volumes/doc/projects/ml/mtcnn_Tensorflow/test_data/next_train/img"
next_train_img_txt = "train_img.txt"


# 转化一个序列列表为稀疏矩阵
def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def decode_sparse_tensor(sparse_tensor):
    # print("sparse_tensor = ", sparse_tensor)
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    # print("decoded_indexes = ", decoded_indexes)
    result = []
    for index in decoded_indexes:
        # print("index = ", index)
        result.append(decode_a_seq(index, sparse_tensor))
        # print(result)
    return result


# 解码
def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = chars[spars_tensor[1][m]]
        decoded.append(str)
    # Replacing blank label to none
    # str_decoded = str_decoded.replace(chr(ord('9') + 1), '')
    # Replacing space label to space
    # str_decoded = str_decoded.replace(chr(ord('0') - 1), ' ')
    # print("ffffffff", str_decoded)
    return decoded


# 生成一个训练batch
def get_next_batch(batch_size=128,ocriter=generator):
    # (batch_size,256,32)
    # FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel
    inputs = np.zeros([batch_size, FLAGS.image_width, FLAGS.image_height])
    codes = []
    for i in range(batch_size):
        # 生成不定长度的字串
        text, image = ocriter.gen_sample()
        # print(image.shape,inputs.shape)
        # np.transpose 矩阵转置 (32*256,) => (32,256) => (256,32)
        inputs[i, :] = np.transpose(image.reshape((FLAGS.image_height, FLAGS.image_width)))
        # inputs[i,:] = image
        codes.append(list(text))
    targets = [np.asarray(i) for i in codes]
    # print(targets)
    sparse_targets = sparse_tuple_from(targets)
    # (batch_size,) 值都是256
    seq_len = np.ones(inputs.shape[0]) * FLAGS.image_width
    return inputs, sparse_targets, seq_len


def get_next_batch_v2(batch_size=128,ocriter=generator):
    return ocriter.get_next_batch_v2(batch_size)

if __name__=="__main__":
    # img = cv2.imread("/Volumes/doc/projects/ml/ocr_tensorflow_cnn/data/train/energy/Jietu20181120-205011.jpg",cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("test",img)
    # cv2.waitKey(0)
    print(np.ceil((3,3)))
