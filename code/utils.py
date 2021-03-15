import matplotlib.pyplot as plt
import numpy as np
import cv2

def utils_plot(img, rows, cols, idx, title, cmap=None, fontsize=20):
    ax = plt.subplot(rows, cols, idx)
    ax.set_title(title, fontsize=fontsize)
    plt.imshow(img, cmap=cmap)
    return
