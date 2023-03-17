import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import cv2
import os
from dataset import Dataset
from model import YOLO_3_SPP_Model
from dataset import imshow

model= YOLO_3_SPP_Model(85)  #实例化yolo v3 SPP model

def make_detect_result():
        """
        在run文件夹下创建detect文件夹，用于存储预测后的图片
        :return:
        """
        folder_name = "detect"
        folder_path = "/Users/qiuhaoxuan/PycharmProjects/深度学习视觉实战/yolo-v3-spp/runs/"
        path = folder_path + folder_name
        files = os.listdir(folder_path)
        if len(files) == 0:
                os.makedirs(path)
        else:
                if files[-1][-1] == 't':
                        path = folder_path + folder_name + '2'
                else:
                        path = folder_path + folder_name + str(int(files[-1][-1])+1)
                os.makedirs(path)
        return path
def make_rectangle(img,k,path,anchor):
        """
        在原图上绘制矩形框并存入run文件夹下的detect文件夹下
        :param img: 图片
        :param anchor: anchor
        :return:
        """
        for i in range(len(anchor)):
                img = cv2.rectangle(img,(anchor[i][0], anchor[i][2]), (anchor[i][2], anchor[i][3]),
                                    color=(0, 255, 0,255), thickness=2)
        cv2.imwrite(os.path.join(path,
                                 "/Users/qiuhaoxuan/PycharmProjects/深度学习视觉实战/yolo-v3-spp/runs/detect/image"+str(k)+".jpg"),img)


def perdict():
        """
        output[0] : torch.Size([1, 85, 7, 7])
        output[0] : torch.Size([1, 85, 14, 14])
        output[0] : torch.Size([1, 85, 28, 28])
        :return:
        """
        data = Dataset("/Users/qiuhaoxuan/PycharmProjects/深度学习视觉实战/yolo-v3-spp/my_yolo_dataset",64,20,224,224)
        data_loader=data.Loader_train()

        # for i,(img,labels,anchor) in enumerate(data_loader):

        path = make_detect_result()
