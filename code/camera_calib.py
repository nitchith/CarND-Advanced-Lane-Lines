import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

def camera_calibrate(images_list, nx=9, ny=6, show_corners=False):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(images_list)
    
    if show_corners:
        fig = plt.figure(figsize=(30, 30))
        rows = 5
        cols = 4

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            if show_corners:
                img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
                ax = plt.subplot(rows, cols, idx + 1)
                ax.set_title(fname)
                plt.imshow(img)

    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[1::-1], None, None)