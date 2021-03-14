import matplotlib.pyplot as plt
import numpy as np
import cv2

def utils_plot(img, rows, cols, idx, title, cmap=None, fontsize=20):
    ax = plt.subplot(rows, cols, idx)
    ax.set_title(title, fontsize=fontsize)
    plt.imshow(img, cmap=cmap)
    return


def thres(img, min_thres, max_thres):
    assert len(img.shape) == 2
    mask = np.zeros_like(img)
    mask[(img >= min_thres) & (img <= max_thres)] = 1
    return mask


def viz_hls(hls):
    fig = plt.figure(figsize=(30, 30))
    utils_plot(hls[:,:,0], 1, 3, 1, "H", "gray")
    utils_plot(hls[:,:,1], 1, 3, 2, "L", "gray")
    utils_plot(hls[:,:,2], 1, 3, 3, "S", "gray")
    return

def viz_masks(rgb, S_mask, mag_mask, dir_mask, final_mask):
    fig = plt.figure(figsize=(30, 30))
    utils_plot(rgb, 3, 2, 1, "Original")
    utils_plot(S_mask, 3, 2, 2, "S_mask", "gray")
    utils_plot(mag_mask, 3, 2, 3, "mag_mask", "gray")
    utils_plot(dir_mask, 3, 2, 4, "dir_mask", "gray")
    utils_plot(final_mask, 3, 2, 5, "final_mask", "gray")
    return    

def viz_result(rgb, mask):
    fig = plt.figure(figsize=(30, 30))
    utils_plot(rgb, 1, 2, 1, "Original")
    utils_plot(mask, 1, 2, 2, "final_mask", "gray")
    return

def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped