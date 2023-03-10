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
    def __init__(self,path, train_batch_size,val_batch_size,width,height):
        super(Dataset, self).__init__()
        self.path = path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.width = width
        self.height = height
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
        label = []
        anchor = []
        images = torch.rand(0)
        label_all = []
        anchor_all = []
        images_all = []
        i=0
        for root, dirs, files in os.walk(self.path):
            break
        if 'train' not in dirs or 'val' not in dirs:
            raise ValueError("Not Found dataset!")
        train_img_path = self.path +'/' +'train' + '/' +'images/'
        train_labels_path = self.path +'/' +'train' + '/' +'labels/'
        for image in os.listdir(train_img_path):
            if image =='.DS_Store':
                continue
            img = cv2.imread(train_img_path + image)
            img = self.resize_image(img,self.width,self.height)
            img = self.to_tensor(img)
            images = torch.cat([images, img],axis=0)
            with open(train_labels_path + image[:-5] + '.txt',"r") as file:
                lines= file.readlines()
            label.append(int(lines[0][:-1]))
            anchor.append(lines[1:])
        label = np.array(label, dtype=np.int32)
        label = torch.tensor(label, dtype=torch.long)
        for i in range(len(anchor)):
            for j in range(len(anchor[i])):
                anchor[i][j] = anchor[i][j].replace("\n","")
                anchor[i][j] = list(map(int, anchor[i][j].split()))
            anchor[i]=np.array(anchor[i], dtype=np.float32)
            anchor[i]= torch.tensor(anchor[i], dtype=torch.float32)
        for i in range(int(round(len(anchor) / self.train_batch_size, 0))):
            if (i + 1) * self.train_batch_size < len(anchor):
                anchor_all.append(anchor[i * self.train_batch_size: (i + 1) * self.train_batch_size])
                images_all.append(images[i * self.train_batch_size: (i + 1) * self.train_batch_size])
                label_all.append(label[i * self.train_batch_size: (i + 1) * self.train_batch_size])
            else:
                anchor_all.append(anchor[i * self.train_batch_size:])
                images_all.append(images[i * self.train_batch_size:])
                label_all.append(label[i * self.train_batch_size:])
        return zip(images_all,label_all,anchor_all)
    def Loader_val(self):
        """
        加载验证集
        :return: (图片,类别,anchoe)
        """
        data_zip=()
        label = []
        anchor = []
        images = torch.rand(0)
        label_all = []
        anchor_all = []
        images_all = []
        i=0
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
            images = torch.cat([images, img],axis=0)
            with open(val_labels_path + image[:-5] + '.txt',"r") as file:
                lines= file.readlines()
            label.append(int(lines[0][:-1]))
            anchor.append(lines[1:])
        label = np.array(label, dtype=np.int32)
        label = torch.tensor(label, dtype=torch.long)
        for i in range(len(anchor)):
            for j in range(len(anchor[i])):
                anchor[i][j] = anchor[i][j].replace("\n","")
                anchor[i][j] = list(map(int, anchor[i][j].split()))
            anchor[i]=np.array(anchor[i], dtype=np.float32)
            anchor[i]= torch.tensor(anchor[i], dtype=torch.float32)
        for i in range(int(round(len(anchor) / self.val_batch_size, 0))):
            if (i + 1) * self.val_batch_size < len(anchor):
                anchor_all.append(anchor[i * self.val_batch_size: (i + 1) * self.val_batch_size])
                images_all.append(images[i * self.val_batch_size: (i + 1) * self.val_batch_size])
                label_all.append(label[i * self.val_batch_size: (i + 1) * self.val_batch_size])
            else:
                anchor_all.append(anchor[i * self.val_batch_size:])
                images_all.append(images[i * self.val_batch_size:])
                label_all.append(label[i * self.val_batch_size:])
        label_all = np.array(label_all, dtype=np.int)
        label_all = torch.tensor(label_all, dtype=torch.long)
        return zip(images_all,label_all,anchor_all)

# label = np.array(labels, dtype=np.int)
# label = torch.tensor(labels, dtype=torch.long)

if __name__ == "__main__":
    import time
    a1=time.time()
    data = Dataset("/Users/qiuhaoxuan/PycharmProjects/深度学习视觉实战/yolo-v3-spp/my_yolo_dataset",64,20,224,224)
    data=data.Loader_train()
    a2=time.time()-a1

    for i,(x,y,z) in enumerate(data):
        break
