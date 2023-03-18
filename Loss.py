import torch
import torch.nn.functional as F
import torch.nn as nn
import math

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

        :param pred_boxes: pred_boxes的shape应该为(batch_size, 4)，其中4表示预测框的左上角和右下角坐标，例如(x1, y1, x2, y2)
        :param target_boxes: pred_boxes的shape应该为(batch_size, 4)，其中4表示预测框的左上角和右下角坐标，例如(x1, y1, x2, y2)
        :return:
        """
        # 计算预测框和目标框的中心点坐标、宽度和高度
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        target_x1, target_y1, target_x2, target_y2 = target_boxes[:, 0], target_boxes[:, 1], \
                                                     target_boxes[:,2], target_boxes[:, 3]
        pred_cx, pred_cy = (pred_x1 + pred_x2) / 2, (pred_y1 + pred_y2) / 2
        target_cx, target_cy = (target_x1 + target_x2) / 2, (target_y1 + target_y2) / 2 #计算中心点
        pred_w, pred_h = pred_x2 - pred_x1, pred_y2 - pred_y1    #计算宽高
        target_w, target_h = target_x2 - target_x1, target_y2 - target_y1   #计算宽高

        # 计算预测框和目标框的面积
        pred_area, target_area = pred_w * pred_h, target_w * target_h

        # 计算预测框和目标框的相交矩形的左上角和右下角坐标
        inter_x1, inter_y1 = torch.max(pred_x1, target_x1), torch.max(pred_y1, target_y1)
        inter_x2, inter_y2 = torch.min(pred_x2, target_x2), torch.min(pred_y2, target_y2)

        # 计算相交矩形的面积和并集矩形的面积
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        union_area = pred_area + target_area - inter_area

        # 计算中心点距离的平方和对角线长度的平方
        center_distance2 = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        diagonal_length2 = (torch.max(pred_x2, target_x2) - torch.min(pred_x1, target_x1)) ** 2 + (
                    torch.max(pred_y2, target_y2) - torch.min(pred_y1, target_y1)) ** 2

        # 计算CIoU Loss
        iou = inter_area / union_area
        v = 4 / (math.pi ** 2) * torch.pow(torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h), 2)
        with torch.no_grad():
            alpha = v / (1 - iou + v)
        ciou_loss = 1 - iou + center_distance2 / diagonal_length2 + alpha * v

        if self.reduction == 'mean':
            return ciou_loss.mean()
        elif self.reduction == 'sum':
            return ciou_loss.sum()
        else:
            return ciou_loss

