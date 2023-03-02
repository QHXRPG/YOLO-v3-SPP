import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import cv2
import os
from model import YOLO_3_SPP_Model

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
def make_rectangle(img,path,x1,y1,x2,y2):
        """
        在原图上绘制矩形框并存入run文件夹下的detect文件夹下
        :param img: 图片
        :param x1: 矩形框左上角坐标x
        :param y1: 矩形框左上角坐标y
        :param x2: 矩形框右下角坐标x
        :param y2: 矩形框右下角坐标y
        :return:
        """
        img = cv2.rectangle(img,(x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.imwrite(os.path.join(path, "image.jpg"), img)

def perdict():
        path = make_detect_result()