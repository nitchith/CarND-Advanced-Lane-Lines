import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def undist_image(img, system):
    return cv2.undistort(img, system.mtx, system.dist, None, system.mtx)

def imread_undist(img_path, system):
    raw = mpimg.imread(img_path)
    undist_img = undist_image(raw, system)
    return undist_img, raw

def plot_raw_undist(raw, undist, fname):
    fig = plt.figure(figsize=(30, 30))
    rows, cols = 1, 2
    ax = plt.subplot(rows, cols, 1)
    ax.set_title("Original Image : {}".format(fname), fontsize=20)
    plt.imshow(raw)
    ax = plt.subplot(rows, cols, 2)
    ax.set_title("Undistorted image".format(fname), fontsize=20)
    plt.imshow(undist)
    return