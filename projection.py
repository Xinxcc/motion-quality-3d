##########################################################################################################
import numpy as np

# Define the intrinsic parameters of the cameras
K1 = np.array([[fx1, 0, cx1],
               [0, fy1, cy1],
               [0, 0, 1]])
K2 = np.array([[fx2, 0, cx2],
               [0, fy2, cy2],
               [0, 0, 1]])

# Define the extrinsic parameters of the cameras
R1 = ...  # rotation matrix of camera 1
t1 = ...  # translation vector of camera 1
R2 = ...  # rotation matrix of camera 2
t2 = ...  # translation vector of camera 2

# Define the perspective transformation matrix
P = K2 @ np.hstack((R2, t2.reshape(-1, 1)))
P = P @ np.linalg.inv(K1 @ np.hstack((R1, t1.reshape(-1, 1))))

# Project the pose landmarks from camera 1 to camera 2
landmarks1 = ...  # 2D pose landmarks from camera 1
homogeneous_landmarks1 = np.hstack((landmarks1, np.ones((len(landmarks1), 1))))
homogeneous_landmarks2 = P @ homogeneous_landmarks1.T
landmarks2 = (homogeneous_landmarks2[:2, :] / homogeneous_landmarks2[2, :]).T

###############################################################################################################


# 求 F####################################################################################################
import cv2
import numpy as np

# Define corresponding points
pts1 = np.array([(xp1, yp1), (xp2, yp2), (xp3, yp3), (xp4, yp4)], dtype=np.float32)
pts2 = np.array(
    [(x_prime_p1, y_prime_p1), (x_prime_p2, y_prime_p2), (x_prime_p3, y_prime_p3), (x_prime_p4, y_prime_p4)],
    dtype=np.float32)

# Compute fundamental matrix using 8-point algorithm
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

# Print fundamental matrix
print("Fundamental Matrix:")
print(F)
###########################################################################################################


# 求E,R,t##################################################################################################
import numpy as np
from numpy.linalg import svd

# Define camera matrices
K1 = np.array([[f1, 0, cx1], [0, f1, cy1], [0, 0, 1]])
K2 = np.array([[f2, 0, cx2], [0, f2, cy2], [0, 0, 1]])

# Compute essential matrix
E = np.matmul(np.matmul(K2.T, F), K1)

# Decompose essential matrix into rotation and translation
U, S, Vt = svd(E)
W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
R1 = np.matmul(np.matmul(U, W), Vt)
R2 = np.matmul(np.matmul(U, W.T), Vt)
t = U[:, 2]

# Print rotation and translation matrices
print("Rotation Matrix 1:")
print(R1)
print("Rotation Matrix 2:")
print(R2)
print("Translation Vector:")
print(t)

#####################################################################################################################


#  求内参数###########################################################################################################

import cv2
import numpy as np

# Load calibration images
image_paths = ['calibration_image1.jpg', 'calibration_image2.jpg', 'calibration_image3.jpg']
images = [cv2.imread(path) for path in image_paths]

# Define calibration board parameters (e.g. chessboard)
board_size = (7, 6)  # Inner corners of the board
square_size = 0.023  # Size of each square in meters (arbitrary value)

# Find chessboard corners in calibration images
object_points = []
image_points = []
for i, image in enumerate(images):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    if ret:
        object_points.append(np.zeros((board_size[0] * board_size[1], 3), np.float32))
        object_points[-1][:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        object_points[-1] *= square_size
        image_points.append(corners)
    else:
        print(f"Failed to find corners in image {i + 1}.")

# Calibrate camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, images[0].shape[:2],
                                                                    None, None)

# Print intrinsic parameters
print("Camera Matrix:")
print(camera_matrix)
print("Distortion Coefficients:")
print(dist_coeffs)

# In this code, we first load a set of calibration images and define the parameters of the calibration board (e.g. a chessboard). We then use the 
# cv2.findChessboardCorners() function to find the chessboard corners in each image, and use them to construct a set of object points and image points.

# We then use the cv2.calibrateCamera() function to calibrate the camera and estimate its intrinsic 
# parameters (camera_matrix) and distortion coefficients (dist_coeffs). Finally, we print out the intrinsic 
# parameters and distortion coefficients.

# Note that this is a basic example and there are many other factors that may need to be considered in real-world 
# scenarios, such as lens distortion models, non-linear optimization, and multiple camera calibration.
############################################################################################################

############################################################################################################
import numpy as np
import cv2
import glob

# Set the dimensions of the checkerboard (number of corners)
CHECKERBOARD = (6, 9)

# Set the termination criteria for the iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points based on the checkerboard dimensions
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Create arrays to store object points and image points from all the images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load the images
images = glob.glob('calibration_images/*.jpg')

# Loop through all the images
for fname in images:
    # Read the image
    img = cv2.imread(fname)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners in the grayscale image
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If the corners are found, add object points and image points
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

# Calibrate the camera using the object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the camera matrix and distortion coefficients to a file
np.savez('calibration_data.npz', mtx=mtx, dist=dist)

# Display the camera matrix and distortion coefficients
print('Camera matrix:\n', mtx)
print('Distortion coefficients:\n', dist)
################################################################################################################

################################################################################################################
import cv2
import numpy as np

# Define the size of the checkerboard in corners
CHECKERBOARD_SIZE = (9, 6)

# Define the size of each square in meters
SQUARE_SIZE = 0.0262  # 26.2 mm

# Load the video file
cap = cv2.VideoCapture('path_to_video_file')

# Arrays to store object points and image points from all the images
obj_points = []  # 3D points in real world space
img_points = []  # 2D points in image plane


# Prepare the object points: (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)

    # If the corners are found, add object points and image points
    if ret == True:
        obj_points.append(objp)
        img_points.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, CHECKERBOARD_SIZE, corners, ret)
        cv2.imshow('frame', frame)
        cv2.waitKey(500)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

# Calibrate the camera and get the camera matrix and distortion coefficients
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1],
                                                                          None, None)

# Print the camera matrix and distortion coefficients
print("Camera matrix:")
print(camera_matrix)
print("Distortion coefficients:")
print(distortion_coeffs)
######################################################################################################

import cv2
import numpy as np
import os
import glob

# Defining the dimensions of checkerboard
CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = glob.glob('./images/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    cv2.imshow('img', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

h, w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)