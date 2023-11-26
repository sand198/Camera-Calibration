'''
Distortion Calibration Algorithm

- First images of chessboard pattern should be captured by wildlife camera traps
- Put all the image in one folder each folder for each camera trap

- https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html 
- https://www.kaggle.com/code/danielwe14/stereocamera-calibration-with-opencv
        '''


import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from mpl_toolkits.axes_grid1 import ImageGrid

SQUARE_SIZE = 26     # how big are the squares of the checkerboard pattern (in millimeters)
BOARD_SIZE = (9,6)  # size of the chessboard (measured from the crossing corners of the squares)

PATH = '../data/chessboard'   #folder path that contain the images of chessboard captured by your camera trap model

print('We have {} Images from the camera'.format(len(os.listdir(PATH))))

# sort the image names after their number
# save the image names with the whole path in a list

print('Before: {}, {}, {}, ...'.format(os.listdir(PATH)[0], os.listdir(PATH)[1], os.listdir(PATH)[2]))

def SortImageNames(path):
    imagelist = sorted(os.listdir(path))
    lengths = []
    for name in imagelist:
        lengths.append(len(name))
    lengths = sorted(list(set(lengths)))
    ImageNames, ImageNamesRaw = [], []
    for l in lengths:
        for name in imagelist:
            if len(name) == l:
                ImageNames.append(os.path.join(path, name))
                ImageNamesRaw.append(name)
    return ImageNames
                
Paths = SortImageNames(PATH)


print('After: {}, {}, {}, ...'.format(os.path.basename(Paths[0]), os.path.basename(Paths[1]), os.path.basename(Paths[2])))

#visualize the image chessboard captured by camera trap as one sample

fig = plt.figure(figsize=(20,20))
grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.1)

for ax, im in zip(grid, [Paths[5]]):
    ax.imshow(plt.imread(im))
    ax.axis('off')

#check if the board size is correct or not

example_image = cv2.imread(Paths[5])
example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2GRAY)

ret, _ = cv2.findChessboardCorners(example_image, BOARD_SIZE)
if ret:
    print('Board Size {} is correct.'.format(BOARD_SIZE))
else:
    print('[ERROR] the Board Size is not correct!')
    BOARD_SIZE = (0,0)

# we have to create the objectpoints
# that are the local 2D-points on the pattern, corresponding 
# to the local coordinate system on the top left corner.

objpoints = np.zeros((BOARD_SIZE[0]*BOARD_SIZE[1], 3), np.float32)
objpoints[:,:2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1,2)
objpoints *= SQUARE_SIZE     

# now we have to find the imagepoints
# these are the same points like the objectpoints but depending
# on the camera coordination system in 3D
# the imagepoints are not the same for each image/camera

def GenerateImagepoints(paths):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    imgpoints = []
    for name in paths:
        img = cv2.imread(name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners1 = cv2.findChessboardCorners(img, BOARD_SIZE)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners1, (4,4), (-1,-1), criteria)
            imgpoints.append(corners2)
    return imgpoints

Left_imgpoints = GenerateImagepoints(Paths)

# Function to calibrate camera
def calibrate_camera(objpoints, imgpoints, img_size):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist

# Calibrate the camera
img = cv2.imread(Paths[0])
img_size = (img.shape[1], img.shape[0])
objpoints = [np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32) for _ in range(len(Left_imgpoints))]
for i in range(len(objpoints)):
    objpoints[i][:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objpoints[i] *= SQUARE_SIZE

# Calibrate the camera
camera_matrix, distortion_coefficients = calibrate_camera(objpoints, Left_imgpoints, img_size)

# Save camera matrix and distortion coefficients to an .npy file
calibration_data = {'CameraMatrix': camera_matrix, 'DistortionCoeffs': distortion_coefficients}
np.save('NPY/distortion_calibration.npy', calibration_data)

print('Camera matrix and distortion coefficients saved to camera_calibration_data.npy')

# we also can display the imagepoints on the example pictures.

def DisplayImagePoints(path, imgpoints):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.drawChessboardCorners(img, BOARD_SIZE, imgpoints, True)
    return img
    
example_image_left = DisplayImagePoints(Paths[1], Left_imgpoints[1])

fig = plt.figure(figsize=(20,20))
grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.1)

for ax, im in zip(grid, [example_image_left]):
    ax.imshow(im)
    ax.axis('off')

# in this picture we now see the local coordinate system of the chessboard
# the origin is at the top left corner
# the orientation is like: long side = X

def PlotLocalCoordinates(img, points):
    points = np.int32(points)
    cv2.arrowedLine(img, tuple(points[0,0]), tuple(points[4,0]), (255,0,0), 3, tipLength=0.05)
    cv2.arrowedLine(img, tuple(points[0,0]), tuple(points[BOARD_SIZE[0]*4,0]), (255,0,0), 3, tipLength=0.05)
    cv2.circle(img, tuple(points[0,0]), 8, (0,255,0), 3)
    cv2.putText(img, '0,0', (points[0,0,0]-35, points[0,0,1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(img, 'X', (points[4,0,0]-25, points[4,0,1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(img, 'Y', (points[BOARD_SIZE[0]*4,0,0]-25, points[BOARD_SIZE[0]*4,0,1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return img

n = 15    ##show the image
img = cv2.imread(Paths[n])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = PlotLocalCoordinates(img, Left_imgpoints[n])

fig = plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis('off')
plt.show()
