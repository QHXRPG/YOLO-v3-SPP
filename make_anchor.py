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
        第一个维度 (32) 表示网格的行数，即将输入图像划分成的网格数量。
        第二个维度 (32) 表示网格的列数，即将输入图像划分成的网格数量。
        第三个维度 (num_anchors) 表示每个网格的 anchor box 数量，即每个网格内生成的 anchor box 的数量。
        第四个维度 (4) 表示每个 anchor box 的左上角和右下角坐标。具体来说，每个 anchor box 的四个坐标值分别表示：
                                                                        左上角的 x 坐标、左上角的 y 坐标、右下角的 x 坐标和右下角的 y 坐标。
        """
        return self.anchors

    def show(self):
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

# 可视化 anchor box
if __name__ == "__main__":
    # 加载图像
    image = Image.open('/Users/qiuhaoxuan/PycharmProjects/yolov3_spp/yolo-v3-spp/my_yolo_dataset/train/images/1.jpeg')
    anchor = Anchor(image, anchor_sizes=[32, 64, 128], anchor_ratios=[0.5, 1, 2], gird_cell_nums=36)
    anchor.built()
    anchor.show()