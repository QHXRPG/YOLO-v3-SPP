import torch
import torch.nn as nn
import numpy as np
import cv2
from Loss import CIoULoss
from make_anchor import Anchor
from model import YOLO_3_SPP_Model
from dataset import Dataset
import torch

def anchor_mask(mask, anchor, grid_num):
    batch_size = mask.size(0)
    result = torch.zeros(batch_size, 4, grid_num, grid_num)

    for i in range(batch_size):
        non_zero_indices = torch.nonzero(mask[i])
        anchor_idx = 0

        for idx in non_zero_indices:
            result[i, :, idx[0], idx[1]] = anchor[i, anchor_idx]
            anchor_idx += 1

    return result

def generate_mask(gt_boxes, image, g):
    # 获取 batch_size
    batch_size = gt_boxes.shape[0]
    # 计算每个网格的宽度和高度，以及每个网格的左上角坐标
    w, h = image.shape[-1], image.shape[-2]
    grid_w, grid_h = w / g, h / g
    grid_x = torch.arange(g).float() * grid_w
    grid_y = torch.arange(g).float() * grid_h
    # 初始化掩码张量
    mask = torch.zeros((batch_size, g, g))
    # 遍历每个样本和每个 ground truth boxes
    for i in range(batch_size):
        for j in range(gt_boxes.shape[1]):
            # 计算中心坐标
            x_center = (gt_boxes[i, j, 0] + gt_boxes[i, j, 2]) / 2
            y_center = (gt_boxes[i, j, 1] + gt_boxes[i, j, 3]) / 2
            # 计算网格索引
            i_index = int(torch.floor(y_center / grid_h))
            j_index = int(torch.floor(x_center / grid_w))
            # 设置掩码值为 1
            mask[i, i_index, j_index] = 1
    for k in range(len(mask)):
        mask[k][0][0] = 0
    return mask

def Prediction_head_info(feature_map:torch.Tensor):
    pred_objectness_probability = torch.sigmoid(feature_map[:,4,:,:])   #存放该网格中是否含有被测物的置信度




#%% 加载数据
import time
data = Dataset("/Users/qiuhaoxuan/PycharmProjects/yolov3_spp/yolo-v3-spp/my_yolo_dataset", 10, 20, 224,
               224)
data_loader = data.Loader_train()
for i, (img, labels, gt_boxes) in enumerate(data_loader):
    break

#%% 前向传播,得到三个特征层
"""
Out[4]: torch.Size([10, 6, 7, 7])  大目标
Out[5]: torch.Size([10, 6, 14, 14]) 中目标
Out[6]: torch.Size([10, 6, 28, 28]) 小目标
"""
model = YOLO_3_SPP_Model(6)
pre = model(img)
feature_map_1,feature_map_2,feature_map_3 = pre[0],pre[1],pre[2]
grid_num1,grid_num2,grid_num3 = feature_map_1.shape[-1], feature_map_2.shape[-1], feature_map_3.shape[-1]
grid_size1,grid_size2,grid_size3 = 224/grid_num1,224/grid_num2,224/grid_num3
#%% 根据生成的三个特征层生成对应的锚框
k=3
image = img[k]
image = image.permute(1, 2, 0).contiguous().numpy()
image = cv2.convertScaleAbs(image)
anchor1,anchor2,anchor3 = Anchor(image, anchor_sizes=[163, 206],
                                 anchor_ratios=[0.5, 1, 2],
                                 gird_cell_nums=grid_num1**2),\
                          Anchor(image, anchor_sizes=[58, 128],
                                 anchor_ratios=[0.5, 1, 2],
                                 gird_cell_nums=grid_num2**2),\
                          Anchor(image, anchor_sizes=[12, 46],
                                 anchor_ratios=[0.5, 1, 2],
                                 gird_cell_nums=grid_num3**2)
anchors1,anchors2,anchors3 = anchor1.built(),anchor2.built(),anchor3.built()

#%% 找到每张图片与ground true(batch_size, 10, 4)  iou最大的锚框 输出anchor_box(batch_size, 10, 4)
# gt_boxes = gt_boxes.numpy()  # 转numpy格式
anchor_box1,anchor_box2,anchor_box3 = anchor1.find_max_iou_anchors(gt_boxes,1),\
                                      anchor2.find_max_iou_anchors(gt_boxes,1),\
                                      anchor3.find_max_iou_anchors(gt_boxes,1)
anchor3.show_all_choice_anchors(anchor_box2[k])


#%% 生成对应的置信度mask
"""
# torch.Size([10, 7, 7])
# torch.Size([10, 14, 14])
# torch.Size([10, 28, 28])
"""
mask1, mask2, mask3 = generate_mask(gt_boxes, img, 7),\
                      generate_mask(gt_boxes, img, 14),\
                      generate_mask(gt_boxes, img, 28)

#%% 定位损失函数计算

anchor1,anchor2,anchor3 = anchor_mask(mask1, anchor_box1, grid_num1),\
                          anchor_mask(mask2, anchor_box2, grid_num2),\
                          anchor_mask(mask3, anchor_box3, grid_num3)
gt1,gt2,gt3 = anchor_mask(mask1, gt_boxes, grid_num1),\
              anchor_mask(mask2, gt_boxes, grid_num2),\
              anchor_mask(mask3, gt_boxes, grid_num3)
gt1[:,2,:,:],gt1[:,3,:,:] = (gt1[:,2,:,:] - gt1[:,0,:,:]),(gt1[:,3,:,:] - gt1[:,1,:,:])
gt2[:,2,:,:],gt2[:,3,:,:] = (gt2[:,2,:,:] - gt2[:,0,:,:]),(gt2[:,3,:,:] - gt2[:,1,:,:])
gt3[:,2,:,:],gt3[:,3,:,:] = (gt3[:,2,:,:] - gt3[:,0,:,:]),(gt3[:,3,:,:] - gt3[:,1,:,:])

e = np.e
Cx1 = mask1*torch.arange(grid_num1)
Cy1 = mask1*torch.arange(grid_num1).reshape(grid_num1,1)
feature_map_1[:,0,:,:] = grid_size1 * torch.sigmoid(feature_map_1[:,0,:,:]) + grid_size1 * Cx1
feature_map_1[:,1,:,:] = grid_size1 * torch.sigmoid(feature_map_1[:,1,:,:]) + grid_size1 * Cy1
feature_map_1[:,2,:,:] = (anchor1[:,2,:,:] - anchor1[:,0,:,:]) * e**(feature_map_1[:,2,:,:])
feature_map_1[:,3,:,:] = (anchor1[:,3,:,:] - anchor1[:,1,:,:]) * e**(feature_map_1[:,3,:,:])

Cx2 = mask2*torch.arange(grid_num2)
Cy2 = mask2*torch.arange(grid_num2).reshape(grid_num2,1)
x2 = grid_size2 * torch.sigmoid(feature_map_2[:,0,:,:]) + grid_size2 * Cx2
y2 = grid_size2 * torch.sigmoid(feature_map_2[:,1,:,:]) + grid_size2 * Cy2
feature_map_2[:,2,:,:] = (anchor2[:,2,:,:] - anchor2[:,0,:,:]) * e**(feature_map_2[:,2,:,:])
feature_map_2[:,3,:,:] = (anchor2[:,3,:,:] - anchor2[:,1,:,:]) * e**(feature_map_2[:,3,:,:])

Cx3 = mask3*torch.arange(grid_num3)
Cy3 = mask3*torch.arange(grid_num3).reshape(grid_num3,1)
x3 = grid_size3 * torch.sigmoid(feature_map_3[:,0,:,:]) + grid_size3 * Cx3
y3 = grid_size3 * torch.sigmoid(feature_map_3[:,1,:,:]) + grid_size3 * Cy3
feature_map_3[:,2,:,:] = (anchor3[:,2,:,:] - anchor3[:,0,:,:]) * e**(feature_map_3[:,2,:,:])
feature_map_3[:,3,:,:] = (anchor3[:,3,:,:] - anchor3[:,1,:,:]) * e**(feature_map_3[:,3,:,:])

ciouloss = CIoULoss(reduction='sum')
l_iou1,l_iou2,l_iou3 = ciouloss(feature_map_1[:,:4,:,:],gt1),\
                       ciouloss(feature_map_2[:,:4,:,:],gt2),\
                       ciouloss(feature_map_3[:,:4,:,:],gt3)



