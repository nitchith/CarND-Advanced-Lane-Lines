import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import utils_plot

def warper(img, system, resize=cv2.INTER_NEAREST):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, system.M, img_size, flags=resize)  # keep same size as input image
    return warped

def plot_orig_warp(orig, warped, cmap=None):
    fig = plt.figure(figsize=(20, 5))
    rows, cols = 1, 2
    ax = plt.subplot(rows, cols, 1)
    ax.set_title("Original Image", fontsize=20)
    plt.imshow(orig, cmap=cmap)
    ax = plt.subplot(rows, cols, 2)
    ax.set_title("Warped Image", fontsize=20)
    plt.imshow(warped, cmap=cmap)
    return