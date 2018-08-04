# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
# 1. 导入各种模块
# 基本形式为：
# import 模块名
# from 某个文件 import 某个模块
# 指定GPU+固定显存
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="3"  # 调用gpu哪块显卡
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 为显卡利用率设置阈值
set_session(tf.Session(config=config))

from PIL import Image  # 图像读写
import numpy as np   # 数组或者矩阵计算
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential   # 新建模型时用到
from keras.layers.advanced_activations import PReLU  # 比较新的激活函数
from keras.optimizers import SGD, Adadelta, Adagrad  # 各种优化器
from keras.utils import np_utils, generic_utils
# from keras.utils.np_utils import to_categorical  # 标签二值化
from keras.utils.vis_utils import plot_model    # 画出网络模型
from six.moves import range
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D,\
concatenate,Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from load_img import load_data    # 加载数据集，返回data，label
from keras.models import Model    # 构建网络模型需要用的模块
from keras import backend as K  # 去除历史计算图
from sklearn.cross_validation import train_test_split  # 用于分类train ：test
np.random.seed(1337)  # for reproducibility
def Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1, 1), name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def Inception(x, nb_filter):

    branch1x1 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch3x3 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch3x3 = Conv2d_BN(branch3x3, nb_filter, (3, 3), padding='same', strides=(1, 1), name=None)
    branch5x5 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch5x5 = Conv2d_BN(branch5x5, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branchpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branchpool = Conv2d_BN(branchpool, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

    x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=3)

    return x


if __name__ == '__main__':

    # 对数据进行预处理
    num_classes = 4  # number of classes
    batch_size = 100
    epochs = 50
    # 加载数据
    if K.image_data_format() == 'channels_first':
        input_shape = (3, 224, 224)  # image shape
    else:
        input_shape = (224, 224, 3)  # image shape
    # load and pre-process data
    def preprocess_input(x):
        return x.astype('float32').reshape((-1,) + input_shape) / 255
    def preprocess_output(y):
        return np_utils.to_categorical(y)

    Data_path = '/home/xxx'
    data, label = load_data(Data_path, 24000)
    label = np_utils.to_categorical(label, 4)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=20)
    # y_train = np_utils.to_categorical(y_train)
    # y_test = np_utils.to_categorical(y_test)
    # x_train, x_test = map(preprocess_input, [x_train, x_test])
    # y_train, y_test = map(preprocess_output,[y_train, y_test])
    print('Loading data...')
    print('x_train shape:', x_train.shape, 'y_train shape:', y_train[1].shape)
    print('x_test shape:', x_test.shape, 'y_test shape', y_test[1].shape)

    input_shape=(224,224,3)
    inputTensor = Input(input_shape)
    # padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
    # model = Sequential()
    x = Conv2d_BN(inputTensor,64, (7, 7), strides=(2, 2), padding='same')
    print('1 x_Conv2d_BN shape:', x.shape)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    print(' x_MaxPooling2D shape:', x.shape)
    x = Conv2d_BN(x, 192, (3, 3), strides=(1, 1), padding='same')
    print('2 x_Conv2d_BN shape:', x.shape)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 64)  # 256
    print('3 x_Inception shape:', x.shape)
    x = Inception(x, 120)  # 480
    print('4 x_Inception shape:', x.shape)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 128)  # 512
    print('5 x_Inception shape:', x.shape)
    x = Inception(x, 128)
    print('6 x_Inception shape:', x.shape)
    x = Inception(x, 128)
    print('7 x_Inception shape:', x.shape)
    x = Inception(x, 132)  # 528
    print('8 x_Inception shape:', x.shape)
    x = Inception(x, 208)  # 832
    print('9 x_Inception shape:', x.shape)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 208)
    print('10 x_Inception shape:', x.shape)
    x = Inception(x, 256)  # 1024
    print('11 x_Inception shape:', x.shape)
    x = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.4)(x)
    print('11 x_Dropout(0.4) shape:', x.shape)
    print('11 x_Flatten shape:', x.shape)
    x = Dense(1000, activation='relu')(x)
    print('12 x_Dense_1 shape:', x.shape)
    x1 = Dense(4, activation='softmax')(x)
    print('13 x_Dense_2 shape:', x1.shape)
    model = Model(input=inputTensor, output=x1)
    # model = Model(inputTensor, x)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # model.fit(data, label, batch_size=32, epochs=1, verbose=1,validation_split=0.2,shuffle=True, initial_epoch=0)
    model.fit(x_train, y_train,batch_size = batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test),shuffle=True)
    # 神经网络模型可视化
    # plot_model(model, to_file='model.png',show_shapes=True)
    model.summary()  # 打印出模型概况，它实际调用的是keras.utils.print_summary
    # 保存模型数据的文件
    save_dir = '/home/xxx/xxx'
    model_name = 'xxx.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    K.clear_session()



