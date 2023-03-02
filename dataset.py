import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import cv2
import os

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

"""图像缩放函数"""
def resize_image(img, width=None, height=None):
    """
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