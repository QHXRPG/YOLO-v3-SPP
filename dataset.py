import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import cv2
import os

classes = ['1', '11', '12', '2', '3', '4']

def imshow(img_name:str,img:np.ndarray):
    cv2.startWindowThread()
    # cv2.imwrite("sp_noise.jpg",noise)  #保存噪声图像
    cv2.imshow(img_name, img)
    # 展示修改后的图片
    while True:
        if ord('q') == cv2.waitKey(0):
            break
    # ord('q')为输出按键q的ASCII码  当与按键输入相等时退出窗口
    cv2.destroyAllWindows()
    cv2.waitKey(1)

class Dataset(Data.Dataset):
    def __init__(self,path, batch_size,width,height):
        super(Dataset, self).__init__()
        self.path = path
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.img = torch.rand(0)
    def to_tensor(self,img):
        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1).contiguous()
        img = img.unsqueeze(0)
        return img
    def resize_image(self,img,width=None, height=None):
        """
        图像缩放函数
        :param img: 图像
        :param width: 宽
        :param height: 高
        :return: 新图像
        """
        h, w = img.shape[:2]
        if width is None and height is None:
            return img
        elif width is None:
            aspect_ratio = float(height) / h
            new_size = (int(w * aspect_ratio), height)
        elif height is None:
            aspect_ratio = float(width) / w
            new_size = (width, int(h * aspect_ratio))
        else:
            new_size = (width, height)
        resized_img = cv2.resize(img, new_size)
        return resized_img
    def Loader_train(self):
        """
        加载训练集
        :return: (图片,类别,anchoe)
        """
        data_zip=()
        for root, dirs, files in os.walk(self.path):
            break
        if 'train' not in dirs or 'val' not in dirs:
            raise ValueError("Not Found dataset!")
        train_img_path = self.path +'/' +'train' + '/' +'images/'
        train_labels_path = self.path +'/' +'train' + '/' +'labels/'
        for image in os.listdir(train_img_path):
            print(image)
            if image =='.DS_Store':
                continue
            img = cv2.imread(train_img_path + image)
            img = self.resize_image(img,self.width,self.height)
            img = self.to_tensor(img)
            with open(train_labels_path + image[:-5] + '.txt',"r") as file:
                lines= file.readlines()
                data_zip = data_zip + ((img, tuple(lines[0])[0], tuple(lines[1:])),)
        return data_zip
    def Loader_val(self):
        """
        加载验证集
        :return: (图片,类别,anchoe)
        """
        data_zip=()
        for root, dirs, files in os.walk(self.path):
            break
        if 'train' not in dirs or 'val' not in dirs:
            raise ValueError("Not Found dataset!")
        val_img_path = self.path +'/' +'val' + '/' +'images/'
        val_labels_path = self.path +'/' +'val' + '/' +'labels/'
        for image in os.listdir(val_img_path):
            if image =='.DS_Store':
                continue
            img = cv2.imread(val_img_path + image)
            img = self.resize_image(img,self.width,self.height)
            img = self.to_tensor(img)
            with open(val_labels_path + image[:-5] + '.txt',"r") as file:
                lines= file.readlines()
                data_zip = data_zip + ((img, tuple(lines[0])[0], tuple(lines[1:])),)
        return data_zip


if __name__ == "__main__":
    data = Dataset("/Users/qiuhaoxuan/PycharmProjects/深度学习视觉实战/yolo-v3-spp/my_yolo_dataset",64,224,224)
    data=data.Loader_train()
