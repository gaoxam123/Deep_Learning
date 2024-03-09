import os
import numpy as np
from PIL import Image
import warnings
from multiprocessing import Pool
from tqdm import tqdm
import cv2

def trim(im):
    percentage = 0.02
    img = np.array(im)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im = img_gray > 0.1 * np.mean(img_gray[img_gray != 0])
    row_sums = np.sum(im, axis=1)
    col_sums = np.sum(im, axis=0)

    rows = np.where(row_sums > img.shape[1] * percentage)[0]
    cols = np.where(col_sums > img.shape[1] * percentage)[0]

    min_row, min_col, max_row, max_col = np.min(rows), np.min(cols), np.max(rows), np.max(cols)
    im_crop = img[min_row:max_row + 1, min_col:max_col + 1]

    return Image.fromarray(im_crop)