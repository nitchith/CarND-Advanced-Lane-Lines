import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import utils_plot

def gen_mask(img, system, show=2):
    ### Parameters ###
    s_min      = system.s_min 
    s_max      = system.s_max 
    mag_min    = system.mag_min 
    mag_max    = system.mag_max 
    mag_kernel = system.mag_kernel 
    dir_min    = system.dir_min 
    dir_max    = system.dir_max 
    dir_kernel = system.dir_kernel 
    ##################
    
    # Cretae a copy of input
    rgb = np.copy(img)
    
    # Threshold on S channel color values
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
    S_mask = thres(hls[:,:,2], s_min, s_max)
    
    # Thereshold on gradient 
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mag_mask = mag_thres(gray, mag_min, mag_max, mag_kernel)
    
    # Threshold on direction
    dir_mask = dir_thres(gray, dir_min, dir_max, dir_kernel)
    
    # Combining all thresholds
    combined_mask = np.zeros_like(gray)
    combined_mask[((S_mask == 1) | (mag_mask == 1)) & (dir_mask == 1) ] = 1
    
    ## plt functions ##
    if show == 1:
        viz_masks(rgb, S_mask, mag_mask, dir_mask, combined_mask)
    elif show == 2:
        viz_result(rgb, combined_mask)
        mpimg.imsave("../output_images/binary.jpg", combined_mask, cmap='gray')
    return combined_mask


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

def mag_thres(gray, min_thres, max_thres, ksize):
    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.sqrt(sobelX**2 + sobelY**2)
    scaled = np.uint8(255*mag/np.max(mag))
    mask = thres(scaled, min_thres, max_thres)
    return mask
    
def dir_thres(gray, min_thres, max_thres, ksize):
    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    sobelX = np.abs(sobelX)
    sobelY = np.abs(sobelY)
    direction = np.arctan2(sobelY, sobelX)
    mask = thres(direction, min_thres, max_thres)
    return mask
    