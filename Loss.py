#%%
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np

def diou_loss(y_true, y_pred):
    """
    计算y_true和y_pred之间的DIoU损失

    Args:
        y_true: Ground truth bounding box (batch_size, 4).
        y_pred: Predicted bounding box (batch_size, 4).

    Returns: diou_loss
    """
    # 提取boxes的坐标
    true_x1, true_y1, true_x2, true_y2 = y_true[:, 0], y_true[:, 1], y_true[:, 2], y_true[:, 3]
    pred_x1, pred_y1, pred_x2, pred_y2 = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]
    # 计算boxes面积
    true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    # 计算交集坐标
    x1 = torch.max(true_x1, pred_x1)
    y1 = torch.max(true_y1, pred_y1)
    x2 = torch.min(true_x2, pred_x2)
    y2 = torch.min(true_y2, pred_y2)
    # 计算交集面积
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    # 计算并集面积
    union = true_area + pred_area - intersection
    # 计算 IoU
    iou = intersection / union
    # 计算两个boxes中心点距离
    true_center_x = (true_x1 + true_x2) / 2
    true_center_y = (true_y1 + true_y2) / 2
    pred_center_x = (pred_x1 + pred_x2) / 2
    pred_center_y = (pred_y1 + pred_y2) / 2
    center_distance = torch.sqrt(torch.pow((true_center_x - pred_center_x), 2) +
                                 torch.pow((true_center_y - pred_center_y), 2))
    # 计算对角线距离
    c_2 = torch.pow((torch.max(true_x2, pred_x2) - torch.min(true_x1, pred_x1)), 2) + \
          torch.pow((torch.max(true_y2, pred_y2) - torch.min(true_y1, pred_y1)), 2)
    d_2 = torch.pow(center_distance, 2) + 1e-7
    diou = iou - (center_distance**2 / d_2) - ((c_2 - d_2) / c_2)
    # 计算批平均值
    diou_loss = 1 - torch.mean(diou)
    return diou_loss



class AlphaLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, gamma=2, logits=True, reduction='mean'):
        super(AlphaLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction
        """
        alpha: 控制正类样本的权重，范围为[0,1]。当alpha=0.5时，表示正类样本和负类样本的权重相等。
        gamma: 控制Focal Loss中的聚焦因子，一般取2。
        logits: 指定输入是否为logits。如果输入为logits，则在计算BCE Loss时需要使用binary_cross_entropy_with_logits函数；
                否则，需要使用binary_cross_entropy函数。
        """
    def forward(self, inputs, targets):
        """
        在forward函数中，我们首先计算二元交叉熵损失BCE_loss。然后，根据BCE_loss计算聚焦因子pt和Focal Loss F_loss。
        最后，根据输入标签targets和超参数alpha，计算每个样本的损失，并根据reduction参数指定的方式进行汇总。
        这个实现可以用于多标签分类任务，其中inputs是模型输出的预测结果，targets是标签。如果每个标签只属于一个类别，
        则可以将该任务视为多类别分类任务，并使用交叉熵损失函数。
        """
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_weight * F_loss

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss



class CIoULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CIoULoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred_boxes, target_boxes):
        """
        :param pred_boxes: pred_boxes的shape应该为(batch_size, 4, gird_nums,gird_nums)，其中4表示预测框的左上角和右下角坐标，例如(x1, y1, w, h)
        :param target_boxes: target_boxes的shape应该为(batch_size, 4, gird_nums,gird_nums)，其中4表示真实框的左上角和右下角坐标，例如(x1, y1, w, h)
        :return:
        """
        batch_size, _, gird_nums, _ = pred_boxes.shape

        # 展平batch和网格尺寸
        pred_boxes = pred_boxes.reshape(batch_size * gird_nums * gird_nums, 4)
        target_boxes = target_boxes.reshape(batch_size * gird_nums * gird_nums, 4)

        # 计算预测和目标框的坐标和尺寸
        pred_x1, pred_y1, pred_w, pred_h = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        target_x1, target_y1, target_w, target_h = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:,2], \
                                                   target_boxes[:, 3]

        # 计算预测和目标框的中心坐标
        pred_cx, pred_cy = pred_x1 + pred_w / 2, pred_y1 + pred_h / 2
        target_cx, target_cy = target_x1 + target_w / 2, target_y1 + target_h / 2
        # 计算预测和目标框的面积
        pred_area, target_area = pred_w * pred_h, target_w * target_h

        # 计算交集和并框的坐标
        inter_x1, inter_y1 = torch.max(pred_x1, target_x1), torch.max(pred_y1, target_y1)
        inter_x2, inter_y2 = torch.min(pred_x1 + pred_w, target_x1 + target_w), torch.min(pred_y1 + pred_h, target_y1 + target_h)
        inter_w, inter_h = torch.clamp(inter_x2 - inter_x1, min=0), torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h
        union_area = pred_area + target_area - inter_area

        # 计算预测框和目标框中心之间的平方距离
        center_distance2 = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

        # 计算最小围框的对角线长度
        diagonal_length2 = torch.max(pred_x1 + pred_w, target_x1 + target_w) - torch.min(pred_x1, target_x1)
        diagonal_length2 += torch.max(pred_y1 + pred_h, target_y1 + target_h) - torch.min(pred_y1, target_y1)
        diagonal_length2 = diagonal_length2 ** 2 + 1e-16

        # 计算 CIoU loss
        iou = inter_area / union_area
        iou[torch.isnan(iou)] = 0
        v = 4 / (math.pi ** 2) * torch.pow(torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h), 2)
        v[torch.isnan(v)] = 0
        with torch.no_grad():
            alpha = v / (1 - iou + v)
        alpha[torch.isnan(alpha)]  = 0
        ciou_loss = 1 - iou + center_distance2 / diagonal_length2 + alpha * v
        ciou_loss[torch.isnan(ciou_loss)] = 0

        # reshape the loss to per-box form
        ciou_loss = ciou_loss.view(batch_size, gird_nums, gird_nums)

        """
        如果reduction设置为'mean'，则返回每个批次的平均CIoU损失值。
        如果reduction设置为'sum'，则返回每个批次的总CIoU损失值。
        如果reduction未设置为'mean'或'sum'，则返回每个批次的未处理CIoU损失值张量。
        """
        if self.reduction == 'mean':
            return ciou_loss.mean()
        elif self.reduction == 'sum':
            return ciou_loss.sum()
        else:
            return ciou_loss



"""
c[0].shape
Out[4]: torch.Size([10, 6, 7, 7])
c[1].shape
Out[5]: torch.Size([10, 6, 14, 14])
c[2].shape
Out[6]: torch.Size([10, 6, 28, 28])
"""
#置信度损失
class ConfidenceLoss(nn.Module):
    def __init__(self):
        super(ConfidenceLoss, self).__init__()

    def forward(self, predictions, targets, object_masks):
        """
        predictions: 模型输出的预测值 (batch_size, num_anchors, num_classes + 5, grid_size, grid_size)
        targets: 真实值 (batch_size, num_anchors, 5, grid_size, grid_size)
        object_masks: 一个与targets维度相同的tensor，用于指示哪些anchor包含目标，哪些不包含目标
        """
        batch_size, num_anchors, _, grid_size, _ = predictions.shape
        num_classes = predictions.shape[2] - 5  # 类别数

        # 将predictions和targets reshape成相同的形状，以便计算损失
        predictions = predictions.view(batch_size, num_anchors, -1, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()
        targets = targets.view(batch_size, num_anchors, -1, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()

        # 获取每个预测框的置信度和实际目标框的IOU，并将其reshape成(batch_size * num_anchors * grid_size * grid_size,)
        objectness_pred = predictions[..., 4]
        objectness_target = targets[..., 4]
        iou_targets = targets[..., 5]
        iou_targets = iou_targets.view(-1)
        object_masks = object_masks.view(-1)

        # 只计算包含目标的预测框的置信度损失
        objectness_loss = nn.BCEWithLogitsLoss(reduction='sum')(objectness_pred[object_masks], objectness_target[object_masks])

        # 将置信度损失除以包含目标的预测框数量的和
        num_objects = torch.sum(object_masks)
        objectness_loss /= num_objects

        return objectness_loss



if __name__ == "__main__":
    #%% 加载数据
    from dataset import Dataset
    import time
    data = Dataset("/Users/qiuhaoxuan/PycharmProjects/yolov3_spp/yolo-v3-spp/my_yolo_dataset", 10, 20, 224,
                   224)
    data_loader = data.Loader_train()
    for i, (img, labels, gt_boxes) in enumerate(data_loader):
        break

    #%% 生成锚框 把一张图片分成36的网格，每个网格生成9个锚框 (6,6,9,4)
    import cv2
    from make_anchor import Anchor
    k=8
    image = img[k]
    image = image.permute(1, 2, 0).contiguous().numpy()
    image = cv2.convertScaleAbs(image)
    anchor = Anchor(image, anchor_sizes=[32, 64, 128], anchor_ratios=[0.5, 1, 2], gird_cell_nums=36)
    anchors = anchor.built()
    #%% 找到每张图片与ground true(batch_size, 10, 4)  iou最大的锚框 输出anchor_box(batch_size, 10, 4)
    gt_boxes = gt_boxes.numpy()  #转numpy格式
    anchor_box = anchor.find_max_iou_anchors(gt_boxes,1)

    #%% 前向传播,得到三个特征层
    """
    Out[4]: torch.Size([10, 6, 7, 7])
    Out[5]: torch.Size([10, 6, 14, 14])
    Out[6]: torch.Size([10, 6, 28, 28])
    """
    from model import YOLO_3_SPP_Model
    model = YOLO_3_SPP_Model(6)
    pre = model(img)
    feature_map_1 = pre[0]
    feature_map_2 = pre[1]
    feature_map_3 = pre[2]

    #%% 损失函数计算
    import torch
    ciouloss = CIoULoss(reduction='sum')
    anchor_box = torch.from_numpy(anchor_box)
    gt_boxes = torch.from_numpy(gt_boxes)
    l = ciouloss(anchor_box,gt_boxes)