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
    def __init__(self,path, train_batch_size,val_batch_size,width,height,max_boxes=10):
        super(Dataset, self).__init__()
        self.path = path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.width = width
        self.height = height
        self.imgname = []
        self.max_boxes = max_boxes

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
        label = []
        target_box = []
        images = torch.rand(0)
        label_all = []
        target_box_all = []
        images_all = []
        self.k = []
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
            self.imgname.append(image)
            h0,w0,_ = img.shape
            self.k.append([h0,w0])
            img = self.resize_image(img,self.width,self.height)
            img = self.to_tensor(img)
            images = torch.cat([images, img],axis=0)
            with open(train_labels_path + image[:-5] + '.txt',"r") as file:
                lines= file.readlines()
            label.append(int(lines[0][:-1]))
            target_box.append(lines[1:])
        label = np.array(label, dtype=np.int32)
        label = torch.tensor(label, dtype=torch.long)
        for i in range(len(target_box)):
            for j in range(len(target_box[i])):
                target_box[i][j] = target_box[i][j].replace("\n", "")
                target_box[i][j] = list(map(int, target_box[i][j].split()))
                target_box[i][j][0] = round(target_box[i][j][0] * self.width / self.k[i][1])
                target_box[i][j][2] = round(target_box[i][j][2] * self.width / self.k[i][1])
                target_box[i][j][1] = round(target_box[i][j][1] * self.height / self.k[i][0])
                target_box[i][j][3] = round(target_box[i][j][3] * self.height / self.k[i][0])
            target_box[i]=np.array(target_box[i], dtype=np.float32)

        gt_boxes_shape = (len(target_box), self.max_boxes, 4)
        gt_boxes_n = np.zeros(gt_boxes_shape)
        target_box = np.array(target_box)
        # 遍历每个二维张量，将目标对象信息复制到全零数组中
        for i in range(self.train_batch_size):
            num_boxes = len(target_box[i])
            if num_boxes > self.max_boxes:
                num_boxes = self.max_boxes
            gt_boxes_n[i, :num_boxes, :] = target_box[i][:self.max_boxes, :]
        gt_boxes_n = torch.from_numpy(gt_boxes_n)

        for i in range(int(round(len(gt_boxes_n) / self.train_batch_size, 0))):
            if (i + 1) * self.train_batch_size < len(gt_boxes_n):
                target_box_all.append(gt_boxes_n[i * self.train_batch_size: (i + 1) * self.train_batch_size])
                images_all.append(images[i * self.train_batch_size: (i + 1) * self.train_batch_size])
                label_all.append(label[i * self.train_batch_size: (i + 1) * self.train_batch_size])
            else:
                target_box_all.append(gt_boxes_n[i * self.train_batch_size:])
                images_all.append(images[i * self.train_batch_size:])
                label_all.append(label[i * self.train_batch_size:])
        return zip(images_all, label_all, target_box_all)

    def Loader_val(self):
        """
        加载验证集
        :return: (图片,类别,anchoe)
        """
        label = []
        target_box = []
        images = torch.rand(0)
        label_all = []
        target_box_all = []
        images_all = []
        self.k = []
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
            target_box.append(lines[1:])
        label = np.array(label, dtype=np.int32)
        label = torch.tensor(label, dtype=torch.long)
        for i in range(len(target_box)):
            for j in range(len(target_box[i])):
                target_box[i][j] = target_box[i][j].replace("\n", "")
                target_box[i][j] = list(map(int, target_box[i][j].split()))
                target_box[i][j][0] = round(target_box[i][j][0] * self.width / self.k[i][1])
                target_box[i][j][2] = round(target_box[i][j][2] * self.width / self.k[i][1])
                target_box[i][j][1] = round(target_box[i][j][1] * self.height / self.k[i][0])
                target_box[i][j][3] = round(target_box[i][j][3] * self.height / self.k[i][0])
            target_box[i]=np.array(target_box[i], dtype=np.float32)
        gt_boxes_shape = (len(target_box), self.max_boxes, 4)
        gt_boxes_n = np.zeros(gt_boxes_shape)
        target_box = np.array(target_box)
        # 遍历每个二维张量，将目标对象信息复制到全零数组中
        for i in range(self.train_batch_size):
            num_boxes = len(target_box[i])
            if num_boxes > self.max_boxes:
                num_boxes = self.max_boxes
            gt_boxes_n[i, :num_boxes, :] = target_box[i][:self.max_boxes, :]
        gt_boxes_n = torch.from_numpy(gt_boxes_n)
        for i in range(int(round(len(gt_boxes_n) / self.train_batch_size, 0))):
            if (i + 1) * self.train_batch_size < len(gt_boxes_n):
                target_box_all.append(gt_boxes_n[i * self.train_batch_size: (i + 1) * self.train_batch_size])
                images_all.append(images[i * self.train_batch_size: (i + 1) * self.train_batch_size])
                label_all.append(label[i * self.train_batch_size: (i + 1) * self.train_batch_size])
            else:
                target_box_all.append(gt_boxes_n[i * self.train_batch_size:])
                images_all.append(images[i * self.train_batch_size:])
                label_all.append(label[i * self.train_batch_size:])
        return zip(images_all, label_all, target_box_all)

def make_Standard_data():
    """
    创建标准数据集
    """
    k = 0
    data = Dataset("/Users/qiuhaoxuan/PycharmProjects/yolov3_spp/yolo-v3-spp/my_yolo_dataset", 64, 20, 224,
                   224)
    data_loader = data.Loader_train()
    for i, (img, labels, anchor) in enumerate(data_loader):
        for j in range(len(img)):
            k = k + 1
            image = img[j]
            image = image.permute(1, 2, 0).contiguous().numpy()
            image = cv2.convertScaleAbs(image)
            for a in range(len(anchor[j])):
                image = cv2.rectangle(image,
                                      (int(anchor[j][a][0]), int(anchor[j][a][1])),
                                      (int(anchor[j][a][2]), int(anchor[j][a][3])),
                                      color=(0, 255, 0, 255), thickness=2)
            cv2.imwrite("/Users/qiuhaoxuan/PycharmProjects/yolov3_spp/yolo-v3-spp/标准数据/image" + str(
                k) + ".jpg", image)


if __name__ == "__main__":
    data = Dataset("/Users/qiuhaoxuan/PycharmProjects/yolov3_spp/yolo-v3-spp/my_yolo_dataset", 64, 20, 224,
                   224)
    data_loader = data.Loader_train()
    for i, (img, labels, gt_boxes) in enumerate(data_loader):
        break