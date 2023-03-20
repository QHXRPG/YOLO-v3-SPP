import numpy as np
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# 计算IoU，矩形框的坐标形式为xywh
def box_iou_xywh(box1, box2):
    x1min, y1min = box1[0] - box1[2]/2.0, box1[1] - box1[3]/2.0
    x1max, y1max = box1[0] + box1[2]/2.0, box1[1] + box1[3]/2.0
    s1 = box1[2] * box1[3]
    x2min, y2min = box2[0] - box2[2]/2.0, box2[1] - box2[3]/2.0
    x2max, y2max = box2[0] + box2[2]/2.0, box2[1] + box2[3]/2.0
    s2 = box2[2] * box2[3]
    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)
    inter_h = np.maximum(ymax - ymin, 0.)
    inter_w = np.maximum(xmax - xmin, 0.)
    intersection = inter_h * inter_w
    union = s1 + s2 - intersection
    iou = intersection / union
    return iou


class Anchor:
    def __init__(self,img, anchor_sizes, anchor_ratios, gird_cell_nums):
        """
        :param img:  输入图像
        :param anchor_sizes:  anchor的所有尺寸
        :param anchor_ratios: anchor所有尺寸的所有长宽比
        :param gird_cell_nums: 划分多少个gird cell
        """
        self.img = img
        self.anchor_sizes = anchor_sizes
        self.anchor_ratios = anchor_ratios
        self.gird_cell_nums = gird_cell_nums
        self.gird_cell_nums_ = int(pow(gird_cell_nums,0.5))
        self.anchor = None
        self.num_anchors = len(self.anchor_sizes) * len(self.anchor_ratios)
        self.isBuilt = False
        self.anchor_box = None
        self.max_ious = None
        self.max_idxs = None
        self.mask = None
        self.anchor_list = None
    def built(self):
        # anchor box 的大小和比例
        # 划分网格并生成 anchor box
        h, w, c = np.array(self.img).shape
        h_i = h // self.gird_cell_nums_  # 一个网格的高度
        w_i = w // self.gird_cell_nums_  # 一个网格的宽度
        self.anchors = np.zeros((self.gird_cell_nums_, self.gird_cell_nums_, self.num_anchors, 4))
        for i in range(self.gird_cell_nums_):
            for j in range(self.gird_cell_nums_):
                # 计算当前网格的左上角和右下角坐标
                x1 = j * w_i
                y1 = i * h_i
                x2 = (j + 1) * w_i
                y2 = (i + 1) * h_i
                # 生成 anchor box
                ctr_x = (x1 + x2) / 2
                ctr_y = (y1 + y2) / 2
                for k in range(len(self.anchor_sizes)):
                    for l in range(len(self.anchor_ratios)):
                        w = self.anchor_sizes[k] * np.sqrt(self.anchor_ratios[l])
                        h = self.anchor_sizes[k] / np.sqrt(self.anchor_ratios[l])
                        self.anchors[i, j, k * len(self.anchor_ratios) + l, 0] = ctr_x - w / 2
                        self.anchors[i, j, k * len(self.anchor_ratios) + l, 1] = ctr_y - h / 2
                        self.anchors[i, j, k * len(self.anchor_ratios) + l, 2] = ctr_x + w / 2
                        self.anchors[i, j, k * len(self.anchor_ratios) + l, 3] = ctr_y + h / 2
        """
        第一个维度 (gird_cell_nums_) 表示网格的行数，即将输入图像划分成的网格数量。
        第二个维度 (gird_cell_nums_) 表示网格的列数，即将输入图像划分成的网格数量。
        第三个维度 (num_anchors) 表示每个网格的 anchor box 数量，即每个网格内生成的 anchor box 的数量。
        第四个维度 (4) 表示每个 anchor box 的左上角和右下角坐标。
        具体来说，每个 anchor box 的四个坐标值分别表示：左上角的 x 坐标、左上角的 y 坐标、右下角的 x 坐标和右下角的 y 坐标。
        """
        self.isBuilt = True
        return self.anchors

    def find_max_iou_anchors(self, gt_boxes, n):
        """
        找到每个 ground truth box 对应的 IOU 最大的n个 anchor

        Args:
        anchors: ndarray, shape为 (h, w, num_anchors, 4)，表示所有anchor box的坐标
        gt_boxes: ndarray, shape为 (batch_size, max_gt_boxes_per_image, 4)，表示所有ground truth box的坐标
        n: int，表示每个ground truth box匹配的最大anchor数量

        Returns:
        max_ious: ndarray, shape为 (batch_size, max_gt_boxes_per_image, num_anchors)，表示每个 ground truth box 和每个 anchor box 的最大IOU值
        max_idxs: ndarray, shape为 (batch_size, max_gt_boxes_per_image, num_anchors)，表示每个 ground truth box 和每个 anchor box 的最大IOU值对应的 anchor box 的索引
        """
        self.mask = np.zeros((gt_boxes.shape[0], gt_boxes.shape[1], self.num_anchors))
        sizes = self.anchor_sizes
        ratios = self.anchor_ratios
        anchors = self.anchors
        image = self.img
        h, w, c = np.array(image).shape
        h_i = h // 6
        w_i = w // 6
        num_anchors = len(sizes) * len(ratios)
        # 找到每个 ground truth box 中心点所在的网格
        gt_ctr_x = (gt_boxes[:, :, 0] + gt_boxes[:, :, 2]) / 2
        gt_ctr_y = (gt_boxes[:, :, 1] + gt_boxes[:, :, 3]) / 2
        gt_grid_x = np.floor(gt_ctr_x / w_i).astype(int)
        gt_grid_y = np.floor(gt_ctr_y / h_i).astype(int)

        # 找到每个 ground truth box 对应的 IOU 最大的 n 个 anchor
        max_ious = np.zeros((gt_boxes.shape[0], gt_boxes.shape[1], num_anchors))
        max_idxs = np.zeros((gt_boxes.shape[0], gt_boxes.shape[1], num_anchors), dtype=np.int32)
        anchor_list = np.zeros((gt_boxes.shape[0], gt_boxes.shape[1], 4))
        for i in range(gt_boxes.shape[0]):
            for j in range(gt_boxes.shape[1]):
                # 只在对应网格内遍历 anchor box
                k = gt_grid_y[i, j]
                l = gt_grid_x[i, j]
                ious = np.zeros(num_anchors)
                for m in range(num_anchors):
                    anchor_box = anchors[k, l, m]
                    # 计算 anchor box 和 ground truth box 的 IOU
                    xx1 = np.maximum(anchor_box[0], gt_boxes[i, j, 0])
                    yy1 = np.maximum(anchor_box[1], gt_boxes[i, j, 1])
                    xx2 = np.minimum(anchor_box[2], gt_boxes[i, j, 2])
                    yy2 = np.minimum(anchor_box[3], gt_boxes[i, j, 3])
                    inter_area = np.maximum(xx2 - xx1, 0) *np.maximum(yy2 - yy1, 0)
                    anchor_box_area = (anchor_box[2] - anchor_box[0]) * (anchor_box[3] - anchor_box[1])
                    gt_box_area = (gt_boxes[i, j, 2] - gt_boxes[i, j, 0]) * (gt_boxes[i, j, 3] - gt_boxes[i, j, 1])
                    iou = inter_area / (anchor_box_area + gt_box_area - inter_area)
                    ious[m] = iou
                    # 找到 IOU 最大的 n 个 anchor
                idxs = np.argsort(-ious)[:n]
                max_ious[i, j, idxs] = ious[idxs]
                max_idxs[i, j, idxs] = k * 6 * 9 + l * 9 + idxs
        #根据max_idxs索引找出相应的iou最大的anchor
        for i in range(max_idxs.shape[0]):
            for j in range(max_idxs.shape[1]):
                for k in range(max_idxs.shape[2]):
                    if max_idxs[i][j][k] >0:
                        anchor_list[i][j]=anchors[max_idxs[i][j][k]//(6*9),
                                                  (max_idxs[i][j][k]%(6*9))//9,
                                                  max_idxs[i][j][k]%9]
        self.anchor_list = anchor_list
        self.max_ious = max_ious
        self.max_idxs = max_idxs
        return self.anchor_list

    def generate_mask(self,ignore_threshold):
        """
        :param ignore_threshold: ignore_threshold: float，表示IoU小于该阈值的anchor box将被视为负样本
        :return: ndarray, shape为 (n, num_anchors)
        """
        # 将IoU大于阈值的anchor box设为正样本，小于阈值的设为负样本
        self.mask[self.max_ious >= ignore_threshold] = 1
        self.mask[self.max_idxs == 1] = 1
        self.mask[self.max_ious < ignore_threshold] = -1
        return self.mask

    def show_all_anchors(self):
        fig, ax = plt.subplots(1)
        ax.imshow(self.img)
        for i in range(self.gird_cell_nums_):
            for j in range(self.gird_cell_nums_):
                for k in range(self.num_anchors):
                    x1 = self.anchors[i, j, k, 0]
                    y1 = self.anchors[i, j, k, 1]
                    x2 = self.anchors[i, j, k, 2]
                    y2 = self.anchors[i, j, k, 3]
                    color = 'r' if k < 3 else 'g' if k < 6 else 'b'
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
        plt.show()
    def show_all_choice_anchors(self, anchor):
        fig, ax = plt.subplots(1)
        ax.imshow(self.img)
        for k in range(len(anchor)):
            x1 = anchor[k][0]
            y1 = anchor[k][1]
            x2 = anchor[k][2]
            y2 = anchor[k][3]
            color = 'r' if k < 1 else 'g' if k < 2 else 'b'
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        plt.show()

# 可视化 anchor box
if __name__ == "__main__":
    #%%
    #加载数据
    from dataset import Dataset
    import time
    data = Dataset("/Users/qiuhaoxuan/PycharmProjects/yolov3_spp/yolo-v3-spp/my_yolo_dataset", 64, 20, 224,
                   224)
    data_loader = data.Loader_train()
    for i, (img, labels, gt_boxes) in enumerate(data_loader):
        break
    #%%
    k=1
    image = img[k]
    image = image.permute(1, 2, 0).contiguous().numpy()
    image = cv2.convertScaleAbs(image)
    anchor = Anchor(image, anchor_sizes=[32, 64, 128], anchor_ratios=[0.5, 1, 2], gird_cell_nums=36)
    anchors = anchor.built()
    #%%
    gt_boxes = gt_boxes.numpy()  #转numpy格式
    a1 = time.time()
    anchor_box = anchor.find_max_iou_anchors(gt_boxes,1)
    a2 = time.time() - a1
    anchor.show_all_choice_anchors(anchor_box[k])
