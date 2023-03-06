import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import cv2
import os
import matplotlib.pyplot as plt
class Image_Transformation():
    def __init__(self,img):
        super(Image_Transformation, self).__init__()
        self.img = img
    #灰度化：将彩色图像转换为灰度图像
    def cvtcolor(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    #对比度调整：增强图像的对比度
    def convertScaleAbs(self,alpha=1.5,beta=0):
        return  cv2.convertScaleAbs(self.img, alpha=alpha, beta=beta)
    #直方图均衡化：将图像的像素值分布拉伸到整个像素值范围内，可以使图像的亮度分布更加均匀
    def equalizeHist(self):
        gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray_image)
    #图像旋转,旋转后的图片大小保持不变
    def getRotationMatrix2D(self,angle=30):
        rows, cols = self.img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)# 计算旋转矩阵
        return cv2.warpAffine(self.img, M, (cols, rows))  # 应用旋转矩阵
    #高斯模糊
    def GaussianBlur(self,ksize=5):
        return cv2.GaussianBlur(self.img, (ksize, ksize), 0)
    #边缘检测
    def Canny(self):
        edges_sobel_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0)  # 在x方向上应用Sobel算子
        return cv2.Canny(self.img, 100, 200)  # 应用Canny边缘检测算法
    #腐蚀操作
    def erode(self,ksize=5):
        kernel = np.ones((ksize, ksize), np.uint8)  # 创建5x5的结构元素
        eroded_image = cv2.erode(self.img, kernel, iterations=1)  # 腐蚀操作
    #膨胀操作
    def dilate(self,ksize=5):
        kernel = np.ones((ksize, ksize), np.uint8)  # 创建5x5的结构元素
        return cv2.dilate(self.img, kernel, iterations=1)  # 膨胀操作
    #开运算
    def morphologyEx_open(self,ksize=5):
        kernel = np.ones((ksize, ksize), np.uint8)  # 创建5x5的结构元素
        return cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel)
    #闭运算
    def morphologyEx_close(self, ksize=5):
        kernel = np.ones((ksize, ksize), np.uint8)  # 创建5x5的结构元素
        return cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)


if  __name__ =='__main__':
    img = cv2.imread("/Users/qiuhaoxuan/PycharmProjects/深度学习视觉实战/yolo-v3-spp/my_yolo_dataset/train/images/1.jpeg")
    it = Image_Transformation(img)
    plt.imshow(it.cvtcolor())
    plt.show()

