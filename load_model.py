# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
# 1. 导入各种模块
# 基本形式为：
# import 模块名
# from 某个文件 import 某个模块
# from .py import def()
# 指定GPU+固定显存
from PIL import Image  # 图像读写
import numpy as np   # 数组或者矩阵计算
from keras.models import load_model
from pan_obj_googlenet_single import create_model
import cv2
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.utils import np_utils, generic_utils #标签二值化
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img


"""------------------数据扩充增强------------------"""
# def my_dada_Generator():
#     datagen=ImageDataGenerator(
#           rotation_range=40,
#           width_shift_range=0.2,
#           height_shift_range=0.2,
#           rescale=1./255,
#           shear_range=0.2,
#           zoom_range=0.2,
#           horizontal_flip=True,
#           fill_mode='nearest')
#     data,label=load_data()
#     for i in datagen.flow(data,batch_size=37,
#                              save_to_dir='/home/4T/wdd/keras_study/20180615/add_data',save_prefix="结婚证_"+i,save_format='jpg'):
#         i+=1
#         if i>30:
#            break
def load_data():
    number = 1100
    label_name = []
    image_path_list = []
    image_label_list = []
    z = 0
    image_path = r'/home/4T/wdd/keras_study/20180615/test_8_27/'
    # 得到类的实际名称
    for i in os.listdir(image_path):
        label_name.append(i)
    # 把所有的类都放在一个列表里,类名下标作为groundtruth
    for i in os.listdir(image_path):

        for j in os.listdir(image_path + '/' + i):
            if '.jpg' in j:
                if z <= 5000000000000000000:
                    dir = image_path + '/' + i + '/' + j
                    image_path_list.append(dir)
                    label = label_name.index(i)
                    image_label_list.append(label)
                    z = z + 1
        z = 0
    num = len(image_label_list)
    print(num)
    data = np.zeros((number, 224, 224, 3), dtype="float32")
    label = np.zeros((number,), dtype=np.uint32)

    i = 0
    for i in range(number - 1):
        print(i)
        img = Image.open(image_path_list[i])
        if img.mode == 'L':
            img = img.convert('RGB')

        if img.mode == '1':
            img = img.convert('RGB')
        print(image_path_list[i])
        img = img.resize((224, 224), Image.ANTIALIAS)
        arr = np.asarray(img, dtype="float32")
        print(img.size, img.mode, img.format)
        data[i, :, :, :] = arr
        # print(image_path_list[i],i)
        # img = load_img(image_path_list[i], target_size=(224, 224), grayscale=False)
        # data[i] = img
        label[i] = int(image_label_list[i])
        # print(label[i])
        with open(r"test_groundtruth.txt","w") as f3:
            for i in range(number - 1):
                f3.write(str(label[i])+","+str(image_path_list[i]))
                f3.write("\n")
        f3.close()
    return data,label


def load_one_data(test_image):
    testImg = Image.open(test_image)
    testImg = testImg.resize((224, 224), Image.ANTIALIAS)
    imgArr = np.asarray(testImg, dtype="float32")
    testData = np.empty((1, 224, 224, 3), dtype="float32")
    testData[0, :, : ,:] = imgArr
    return testData


if __name__ == '__main__':
    # my_dada_Generator()
    # model = create_model(dropout=0.4)
    # test_image = r"/home/4T/wdd/keras_study/20180615/test_8_27/身份证（已满1000张）/身份证 (2718).jpg"
    # src=cv2.imread(test_image)
    # src=cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
    # cv2.imshow("test image: ",src)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    """------------------加载模型------------------"""
    model_path="/home/4T/wdd/keras_study/For_20180615/model/Certificate_googlenet_9000_B100_epoch50_0.998.h5"
    model=load_model(model_path)

    # """------------------测试单张图像------------------"""
    # testData=load_one_data(test_image)
    # predict = model.predict(testData, batch_size=1, verbose=1)
    # print(predict)
    # result = np.argmax(predict)
    # print('待测的图片最大的概率为 : ', np.max(predict))
    # print('待测的图片类别是 : ', result)
    #
    # np.savetxt('test_predict.txt', result, delimiter=',')
    """------------------测试多张图像------------------"""
    data,label=load_data()
    predict_P=model.predict(data, batch_size=1100, verbose=1)
    label = np_utils.to_categorical(label, 9)
    print(predict_P)
    result=[]
    pred_list = []

    with open("test_predict.txt","w") as f6:
        for i in range(len(predict_P)):
            print('待测的图片是 : ',np.argmax(predict_P[i]))
            pred_list.append(np.argsort(predict_P[i])) # 获取最大的N个值的下标
            result.append(np.argmax(predict_P[i]))
            f6.write(str(result[i])+","+str(np.max(predict_P[i])))
            f6.write(","+str(np.argsort(predict_P[i])))

            f6.write("\n")
    f6.close()

    # """------------------测试准确率------------------"""
    # num_success = 0
    # num_failed = 0
    # predict=[]
    # groundtruth=[]
    # with open(r"test_groundtruth.txt", 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         line = line.strip("\n")
    #         predict.append(line.split(",")[2])
    #         groundtruth.append(line.split(",")[0])
    #         if line.split(",")[-1]==line.split(",")[0]:
    #             num_success = num_success + 1
    #         else:
    #             num_failed = num_failed + 1
    #     print("predict: ", predict)
    #     print("groundtruth: ", groundtruth)
    #     print(len(groundtruth), num_success, num_failed)
    #     result_success = 1.0 * num_success / len(predict)
    #     result_failed = 1.0 * num_failed / len(predict)
    #     print("result_success: %s " % result_success)
    #     print("result_failed: %s " % result_failed)
    #
    # f.close()
    """------------------top1 准确率及每类准确率------------------"""
    print('----------------------------------------------------')
    N = 1
    i=0
    num_classes=9
    pred_list = []
    for row in predict_P:
        pred_list.append(row.argsort()[-N:][::-1])  # 获取最大的N个值的下标
    pred_array = np.array(pred_list)
    test_arg = np.argmax(label, axis=1)
    class_count = [0 for _ in range(num_classes)]
    class_acc = [0 for _ in range(num_classes)]
    for i in range(len(test_arg)):
        class_count[test_arg[i]] += 1
        if test_arg[i] in pred_array[i]:
            class_acc[test_arg[i]] += 1
    print('top-' + str(N) + ' all acc:', str(sum(class_acc)) + '/' + str(len(test_arg)),
          sum(class_acc) / float(len(test_arg)))

    for i in range(num_classes):
        print(i, 'acc: ' + str(class_acc[i]) + '/' + str(class_count[i]))

    # """------------------topN 准确率及每类准确率------------------"""
    # print('----------------------------------------------------')
    # N = 1
    # i = 0
    # num_classes = 9
    # pred_list = []
    # for row in predict_P:
    #     pred_list.append(row.argsort()[-N:][::-1])  # 获取最大的N个值的下标
    # pred_array = np.array(pred_list)
    # test_arg = np.argmax(label, axis=1)
    # class_count = [0 for _ in range(num_classes)]
    # class_acc = [0 for _ in range(num_classes)]
    # # 对每一类计算准确率
    # for i in range(len(test_arg)):
    #     class_count[test_arg[i]] += 1
    #
    #     if test_arg[i] in pred_array[i]:
    #         class_acc[test_arg[i]] += 1
    #
    # print('top-' + str(N) + ' all acc:', str(sum(class_acc)) + '/' + str(len(test_arg)),
    #       sum(class_acc) / float(len(test_arg)))
    #
    # for i in range(num_classes):
    #     print(i, 'acc: ' + str(class_acc[i]) + '/' + str(class_count[i]))
