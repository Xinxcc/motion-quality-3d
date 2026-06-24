import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import svd
from scipy.signal import correlate, resample_poly, correlation_lags
from scipy import interpolate
from scipy.interpolate import interp1d
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
import csv
from scipy.stats import zscore

# 计算失真系数，相机矩阵


def camera_calibration(path,
                       checkboard_size=(9, 6),
                       square_size=0.0262,
                       show=False,
                       save=False,
                       frames=[]):
    # Define the number of inner corners of the checkerboard
    CHECKERBOARD_SIZE = checkboard_size
    # Define the size of each square in meters
    SQUARE_SIZE = square_size  # 26.2 mm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Load the video file
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    # Arrays to store object points and image points from all the images
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane
    # Prepare the object points: (0,0,0), (1,0,0), ..., (8,5,0)
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3),
                    np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0],
                           0:CHECKERBOARD_SIZE[1]].T.reshape(-1,
                                                             2) * SQUARE_SIZE
    # Create a new video writer to write the undistorted frames
    if save:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_name = path[:-4] + "_findboard.avi"
        out = cv2.VideoWriter(output_name,
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                              (width, height))
    while cap.isOpened():
        if len(frames) > 0 and frame_count not in frames:
            if len(frames) > 0 and frame_count > frames[-1]:
                print("End of selected segment reached!")
                break
            frame_count = frame_count + 1
            cap.grab()
            pass
        else:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("End of video reached!")
                break
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(
                gray, CHECKERBOARD_SIZE, flags=cv2.CALIB_CB_FAST_CHECK
            )  # cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            # If the corners are found, add object points and image points
            if ret:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1),
                                            criteria)
                img_points.append(corners2)
            else:
                print("No checkerboard found!")
                pass
            if save:
                if ret:
                    cv2.drawChessboardCorners(frame, CHECKERBOARD_SIZE,
                                              corners2, ret)
                out.write(frame)
            if show:
                if ret:
                    cv2.drawChessboardCorners(frame, CHECKERBOARD_SIZE,
                                              corners2, ret)
                frame = cv2.resize(frame, (1280, 960))
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Manual Interrupt!")
                    break
            frame_count = frame_count + 1
    # Release the video capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
    # Calibrate the camera and get the camera matrix and distortion coefficients
    ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None)
    # Print the camera matrix and distortion coefficients
    print("Camera matrix:")
    print(camera_matrix)
    print("Distortion coefficients:")
    print(distortion_coeffs)
    return camera_matrix, distortion_coeffs


def undistort_video(path,
                    camera_matrix,
                    dist_coeffs,
                    show=True,
                    save=False,
                    start_frame=0,
                    end_frame=int):
    # for 2560*1920
    # Load the video file
    cap = cv2.VideoCapture(path)
    frame_count = 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if save:
        output_name = path[:-5] + "_undistorted.avi"
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_name,
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                              (2394, 1684))  # (2394,1684)
    while cap.isOpened():
        if frame_count < start_frame:
            frame_count = frame_count + 1
            cap.grab()
            pass
        elif frame_count > end_frame:
            print("End of selected segment reached!")
            break
        else:
            # Capture frame-by-frameq
            ret, frame = cap.read()
            if not ret:
                print("End of video reached!")
                break
            new_mtx, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (width, height), 0.7,
                (width, height))
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs,
                                        None, new_mtx)
            undistorted = undistorted[roi[1]:roi[1] + roi[3],
                                      roi[0]:roi[0] + roi[2]]
            if save:
                out.write(undistorted)
            if show:
                cv2.imshow('frame', undistorted)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Manual Interrupt!")
                    break
            frame_count = frame_count + 1

    # Release the video capture and writer objects
    cap.release()
    cv2.destroyAllWindows()
    return output_name


def frame_diff(index, path_camera, path_sensor, worldmarks=False, save=False):

    df = pd.read_csv(path_camera)
    video_name_list = df['filename'].unique()
    # [Biceps,Cor_Squat,Far_Squat,Rotation_lef,Slight_Squat]
    
    df1 = df.loc[df['filename'] == video_name_list[index]]
    df2 = df.loc[df['filename'] == video_name_list[index+1]]
    df3 = pd.read_csv(path_sensor)
    # cut off first
    df1, df2, df3 = cut_dataframe(df1, df2, df3, index)

    if worldmarks == False:
        axis_mp = 'y'
        axis_sens = 'z'
    else:
        axis_mp = 'z'
        axis_sens = 'z'
    df1_diff = df1.iloc[:, df1.columns.str.endswith('.' + axis_mp)].diff(
        periods=1, axis=0).sum(axis=1)
    df2_diff = df2.iloc[:, df2.columns.str.endswith('.' + axis_mp)].diff(
        periods=1, axis=0).sum(axis=1)
    df3_diff = df3.iloc[:, df3.columns.str.endswith(' ' + axis_sens)].diff(
        periods=1, axis=0).sum(axis=1)
    df1_diff = gaussian_filter1d(
        (df1_diff-np.min(df1_diff))/(np.max(df1_diff)-np.min(df1_diff)), sigma=3)
    df2_diff = gaussian_filter1d(
        (df2_diff-np.min(df2_diff))/(np.max(df2_diff)-np.min(df2_diff)), sigma=3)
    df3_diff = gaussian_filter1d(
        (df3_diff-np.min(df3_diff))/(np.max(df3_diff)-np.min(df3_diff)), sigma=3)
    # df1_diff=(df1_diff-np.min(df1_diff))/(np.max(df1_diff)-np.min(df1_diff))
    # df2_diff=(df2_diff-np.min(df2_diff))/(np.max(df2_diff)-np.min(df2_diff))
    # df3_diff=(df3_diff-np.min(df3_diff))/(np.max(df3_diff)-np.min(df3_diff))
    x1_resampled = resample_poly(df1_diff, 60, 30)
    x2_resampled = resample_poly(df2_diff, 60, 30)
    x3_resampled = resample_poly(df3_diff, 60, 60)

    t1 = np.arange(x1_resampled.shape[0]) / 60.0  # Assuming 60 fps sample rate
    t2 = np.arange(x2_resampled.shape[0]) / 60.0
    t3 = np.arange(x3_resampled.shape[0]) / 60.0

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 9))
    axs[0].plot(t1, x1_resampled, label='camera left')
    # axs[0].set_ylabel('diff alone {} axis'.format(axis_mp))
    axs[0].set_title('Before')
    axs[0].legend()
    axs[1].plot(t2, x2_resampled, label='camera right')
    # axs[1].set_ylabel('diff alone {} axis'.format(axis_mp))
    axs[1].legend()
    axs[2].plot(t3, 1-x3_resampled, label='sensor')
    # axs[2].set_ylabel('diff alone {} axis'.format(axis_sens))
    axs[2].legend()
    # plt.savefig('diff_before.jpg')
    plt.show()

    corr12 = correlate(x2_resampled, x1_resampled, mode='full')
    shift12 = np.argmax(corr12) - (len(x1_resampled) - 1)
    corr13 = correlate(1-x3_resampled, x1_resampled, mode='full')
    shift13 = np.argmax(corr13) - (len(x1_resampled) - 1)
    corr23 = correlate(1-x3_resampled, x2_resampled, mode='full')
    shift23 = np.argmax(corr23) - (len(x2_resampled) - 1)

    # 平移
    if shift13 >= 0:
        if shift23 >= 0:
            if shift13 >= shift23:
                x3_aligned = x3_resampled[shift13:]
                x2_aligned = x2_resampled[shift13 - shift23:]
                x1_aligned = x1_resampled[:min(len(x2_aligned), len(x3_aligned)
                                               )]
            else:
                x3_aligned = x3_resampled[shift23:]
                x1_aligned = x1_resampled[shift23 - shift13:]
                x2_aligned = x2_resampled[:min(len(x1_aligned), len(x3_aligned)
                                               )]
        elif shift23 < 0:
            x3_aligned = x3_resampled[shift13:]
            x2_aligned = x2_resampled[shift13 - shift23:]
            x1_aligned = x1_resampled[:min(len(x2_aligned), len(x3_aligned))]
    elif shift13 < 0:
        if shift23 >= 0:
            x3_aligned = x3_resampled[shift23:]
            x1_aligned = x1_resampled[shift23 - shift13:]
            x2_aligned = x2_resampled[:min(len(x1_aligned), len(x3_aligned))]
        elif shift23 < 0:
            if shift13 >= shift23:
                x2_aligned = x2_resampled[-shift23:]
                x1_aligned = x1_resampled[-shift13:]
                x3_aligned = x3_resampled[:min(len(x1_aligned), len(x2_aligned)
                                               )]
            else:
                x1_aligned = x1_resampled[-shift13:]
                x2_aligned = x2_resampled[-shift23:]
                x3_aligned = x3_resampled[:min(len(x1_aligned), len(x2_aligned)
                                               )]

    # Trim the signals to the same length if necessary
    length = min(len(x1_aligned), len(x2_aligned), len(x3_aligned))

    x1_aligned = x1_aligned[:length]
    x2_aligned = x2_aligned[:length]
    x3_aligned = x3_aligned[:length]

    df1, df2, df3 = syn_dataframe(df1, df2, df3, shift13, shift23)

    fig, axs = plt.subplots(nrows=3,
                            ncols=1,
                            figsize=(12, 9),
                            sharex=True,
                            sharey=True)
    # fig.suptitle('synchronisation by using {} '.format(NAME[i]))
    t4 = np.arange(x1_aligned.shape[0]) / 60.0
    axs[0].plot(t4, x1_aligned, label='camera left')
    # axs[0].set_ylabel('diff alone {} axis'.format(axis_mp))
    axs[0].set_title('After')
    axs[0].legend()
    axs[1].plot(t4, x2_aligned, label='camera right')
    # axs[1].set_ylabel('diff alone {} axis'.format(axis_mp))
    axs[1].legend()
    axs[2].plot(t4, 1-x3_aligned, label='sensor')
    # axs[2].set_ylabel('diff alone {} axis'.format(axis_sens))
    axs[2].legend()
    # plt.savefig('diff_after1.jpg')
    plt.show()

    return shift13, shift23, shift12


def rot_trans_sen(index, path_sensor, path_camera, cam, dist, shift13, shift23):

    size = np.array([2394, 1684])
    df = pd.read_csv(path_camera)
    video_name_list = df['filename'].unique()
    # [Biceps,Cor_Squat,Far_Squat,Rotation_lef,Slight_Squat]
    df1 = df.loc[df['filename'] == video_name_list[index]]
    df2 = df.loc[df['filename'] == video_name_list[index+1]]
    df3 = pd.read_csv(path_sensor)
    # cut off first
    df1, df2, df3 = cut_dataframe(df1, df2, df3, index)
    
    df1, df2, df3 = syn_dataframe(df1, df2, df3, shift13, shift23)
    # left_arm = [12, 13, 14]
    # right_arm = [8, 9, 10]
    # left_leg = [19, 20, 21]
    # right_leg = [15, 16, 17]
    limbs = [12, 13, 14, 8, 9, 10, 19, 20, 21, 15, 16, 17]

    points_sensor = np.array([[df3.iloc[:, (limbs[0])*3+1], df3.iloc[:, (limbs[0])*3+2], df3.iloc[:, (limbs[0])*3+3]],
                              [df3.iloc[:, (limbs[1])*3+1], df3.iloc[:,
                                                                     (limbs[1])*3+2], df3.iloc[:, (limbs[1])*3+3]],
                              [df3.iloc[:, (limbs[2])*3+1], df3.iloc[:,
                                                                     (limbs[2])*3+2], df3.iloc[:, (limbs[2])*3+3]],
                              [df3.iloc[:, (limbs[3])*3+1], df3.iloc[:, (limbs[3])*3+2],
                               df3.iloc[:, (limbs[3])*3+3]],
                              [df3.iloc[:, (limbs[4])*3+1], df3.iloc[:, (limbs[4])*3+2],
                               df3.iloc[:, (limbs[4])*3+3]],
                              [df3.iloc[:, (limbs[5])*3+1], df3.iloc[:, (limbs[5])*3+2],
                               df3.iloc[:, (limbs[5])*3+3]],
                              [df3.iloc[:, (limbs[6])*3+1], df3.iloc[:, (limbs[6])*3+2],
                               df3.iloc[:, (limbs[6])*3+3]],
                              [df3.iloc[:, (limbs[7])*3+1], df3.iloc[:, (limbs[7])*3+2],
                               df3.iloc[:, (limbs[7])*3+3]],
                              [df3.iloc[:, (limbs[8])*3+1], df3.iloc[:, (limbs[8])*3+2],
                               df3.iloc[:, (limbs[8])*3+3]],
                              [df3.iloc[:, (limbs[9])*3+1], df3.iloc[:, (limbs[9])*3+2],
                               df3.iloc[:, (limbs[9])*3+3]],
                              [df3.iloc[:, (limbs[10])*3+1], df3.iloc[:, (limbs[10])
                                                                      * 3+2], df3.iloc[:, (limbs[10])*3+3]],
                              [df3.iloc[:, (limbs[11])*3+1], df3.iloc[:, (limbs[11])*3+2], df3.iloc[:, (limbs[11])*3+3]]
                            ])
    pts_3d = np.hstack(points_sensor).T

    joints = [11, 13, 15, 12, 14, 16, 23, 25, 27, 24, 26, 28]
    points_camera_left = np.array([[df1.iloc[:, (joints[0])*4+2], df1.iloc[:, (joints[0])*4+3]],
                                   [df1.iloc[:, (joints[1])*4+2],
                                    df1.iloc[:, (joints[1])*4+3]],
                                   [df1.iloc[:, (joints[2])*4+2],
                                    df1.iloc[:, (joints[2])*4+3]],
                                   [df1.iloc[:, (joints[3])*4+2],
                                    df1.iloc[:, (joints[3])*4+3]],
                                   [df1.iloc[:, (joints[4])*4+2],
                                    df1.iloc[:, (joints[4])*4+3]],
                                   [df1.iloc[:, (joints[5])*4+2],
                                    df1.iloc[:, (joints[5])*4+3]],
                                   [df1.iloc[:, (joints[6])*4+2],
                                    df1.iloc[:, (joints[6])*4+3]],
                                   [df1.iloc[:, (joints[7])*4+2],
                                    df1.iloc[:, (joints[7])*4+3]],
                                   [df1.iloc[:, (joints[8])*4+2],
                                    df1.iloc[:, (joints[8])*4+3]],
                                   [df1.iloc[:, (joints[9])*4+2],
                                    df1.iloc[:, (joints[9])*4+3]],
                                   [df1.iloc[:, (joints[10])*4+2],
                                    df1.iloc[:, (joints[10])*4+3]],
                                   [df1.iloc[:, (joints[11])*4+2], df1.iloc[:, (joints[11])*4+3]]
                                ])
    pts_2d_left = np.hstack(points_camera_left).T * size

    points_camera_right = np.array([[df2.iloc[:, (joints[0])*4+2], df2.iloc[:, (joints[0])*4+3]],
                                    [df2.iloc[:, (joints[1])*4+2],
                                     df2.iloc[:, (joints[1])*4+3]],
                                    [df2.iloc[:, (joints[2])*4+2],
                                     df2.iloc[:, (joints[2])*4+3]],
                                    [df2.iloc[:, (joints[3])*4+2],
                                     df2.iloc[:, (joints[3])*4+3]],
                                    [df2.iloc[:, (joints[4])*4+2],
                                     df2.iloc[:, (joints[4])*4+3]],
                                    [df2.iloc[:, (joints[5])*4+2],
                                     df2.iloc[:, (joints[5])*4+3]],
                                    [df2.iloc[:, (joints[6])*4+2],
                                     df2.iloc[:, (joints[6])*4+3]],
                                    [df2.iloc[:, (joints[7])*4+2],
                                     df2.iloc[:, (joints[7])*4+3]],
                                    [df2.iloc[:, (joints[8])*4+2],
                                     df2.iloc[:, (joints[8])*4+3]],
                                    [df2.iloc[:, (joints[9])*4+2],
                                     df2.iloc[:, (joints[9])*4+3]],
                                    [df2.iloc[:, (joints[10])*4+2],
                                     df2.iloc[:, (joints[10])*4+3]],
                                    [df2.iloc[:, (joints[11])*4+2], df2.iloc[:, (joints[11])*4+3]]
                                    ])
    pts_2d_right = np.hstack(points_camera_right).T * size
    pts_2d_left_resample = resample_poly(pts_2d_left, 60, 30)
    pts_2d_right_resample = resample_poly(pts_2d_right, 60, 30)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pts_3d.T[0], pts_3d.T[1], pts_3d.T[2], marker='.')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('3d_sensor')
    # plt.savefig('3d_sensor.jpg')
    plt.show()
    retval_l, rvec_left, tvec_left = cv2.solvePnP(
        pts_3d, pts_2d_left_resample, cam, dist)
    retval_r, rvec_right, tvec_right = cv2.solvePnP(
        pts_3d, pts_2d_right_resample, cam, dist)
    P_left = rtvec_to_matrix(rvec_left, tvec_left)
    P_right = rtvec_to_matrix(rvec_right, tvec_right)
    return pts_3d, P_left, P_right, rvec_left, tvec_left, rvec_right, tvec_right


def rot_trans_cam(index, path_camera, path_sensor, K, dist, P, shift13, shift23):

    size = np.array([2394, 1684])
    df = pd.read_csv(path_camera)
    video_name_list = df['filename'].unique()
    # [Biceps,Cor_Squat,Far_Squat,Rotation_lef,Slight_Squat]
    # index2 df1[200:-150] df2[200:-150]
    # index8 df1 = df1[130:-300] df2 = df2[250:-140] df3 = df3[720:-438]
    df1 = df.loc[df['filename'] == video_name_list[index]]
    df2 = df.loc[df['filename'] == video_name_list[index+1]]
    df3 = pd.read_csv(path_sensor)
    # cut off first
    df1, df2, df3 = cut_dataframe(df1, df2, df3, index)
    # df3 = df3[47200:48720]
    df1, df2, df3 = syn_dataframe(df1, df2, df3, shift13, shift23)

    size = np.array([2394, 1684])

    # size = np.array([2560, 1920])
    joints = [11, 13, 15, 12, 14, 16, 23, 25, 27, 24, 26, 28]
    points_left = np.array([[df1.iloc[:, (joints[0])*4+2], df1.iloc[:, (joints[0])*4+3]],
                            [df1.iloc[:, (joints[1])*4+2],
                             df1.iloc[:, (joints[1])*4+3]],
                            [df1.iloc[:, (joints[2])*4+2],
                             df1.iloc[:, (joints[2])*4+3]],
                            [df1.iloc[:, (joints[3])*4+2],
                             df1.iloc[:, (joints[3])*4+3]],
                            [df1.iloc[:, (joints[4])*4+2],
                             df1.iloc[:, (joints[4])*4+3]],
                            [df1.iloc[:, (joints[5])*4+2],
                             df1.iloc[:, (joints[5])*4+3]],
                            [df1.iloc[:, (joints[6])*4+2],
                             df1.iloc[:, (joints[6])*4+3]],
                            [df1.iloc[:, (joints[7])*4+2],
                             df1.iloc[:, (joints[7])*4+3]],
                            [df1.iloc[:, (joints[8])*4+2],
                             df1.iloc[:, (joints[8])*4+3]],
                            [df1.iloc[:, (joints[9])*4+2],
                             df1.iloc[:, (joints[9])*4+3]],
                            [df1.iloc[:, (joints[10])*4+2],
                             df1.iloc[:, (joints[10])*4+3]],
                            [df1.iloc[:, (joints[11])*4+2], df1.iloc[:, (joints[11])*4+3]]
                            ])
    pts_left = np.hstack(points_left).T * size
    points_right = np.array([[df2.iloc[:, (joints[0])*4+2], df2.iloc[:, (joints[0])*4+3]],
                             [df2.iloc[:, (joints[1])*4+2],
                              df2.iloc[:, (joints[1])*4+3]],
                             [df2.iloc[:, (joints[2])*4+2],
                              df2.iloc[:, (joints[2])*4+3]],
                             [df2.iloc[:, (joints[3])*4+2],
                              df2.iloc[:, (joints[3])*4+3]],
                             [df2.iloc[:, (joints[4])*4+2],
                              df2.iloc[:, (joints[4])*4+3]],
                             [df2.iloc[:, (joints[5])*4+2],
                              df2.iloc[:, (joints[5])*4+3]],
                             [df2.iloc[:, (joints[6])*4+2],
                              df2.iloc[:, (joints[6])*4+3]],
                             [df2.iloc[:, (joints[7])*4+2],
                              df2.iloc[:, (joints[7])*4+3]],
                             [df2.iloc[:, (joints[8])*4+2],
                              df2.iloc[:, (joints[8])*4+3]],
                             [df2.iloc[:, (joints[9])*4+2],
                              df2.iloc[:, (joints[9])*4+3]],
                             [df2.iloc[:, (joints[10])*4+2],
                              df2.iloc[:, (joints[10])*4+3]],
                             [df2.iloc[:, (joints[11])*4+2], df2.iloc[:, (joints[11])*4+3]]
                            ])
    pts_right = np.hstack(points_right).T * size
    K = K
    dist_coef = dist

    # Normalize image coordinates of corresponding points
    # pts1_norm = cv2.undistortPoints(pts1, K, dist_coef)
    # pts1_norm = pts1_norm.reshape(-1, 2)
    # pts2_norm = cv2.undistortPoints(pts2, K, dist_coef)
    # pts2_norm = pts2_norm.reshape(-1, 2)

    # Compute fundamental matrix from normalized points
    F, mask1 = cv2.findFundamentalMat(
        pts_left, pts_right, cv2.FM_8POINT)
    # F, mask1 = cv2.findFundamentalMat(
    #     pts_left, pts_right, method=cv2.FM_RANSAC)
    # F, mask1 = cv2.findFundamentalMat(
    #     pts_left, pts_right, cv2.FM_LMEDS)

    # Compute essential matrix from fundamental matrix and intrinsic matrix
    E = K.T @ F @ K

    # p1, p2 = cv2.correctMatches(E, pts1.T, pts2.T)

    retval, R, t, mask = cv2.recoverPose(E, pts_left, pts_right, K)
    mask = mask.astype('uint8')
    pts1_filtered = pts_left[mask.ravel() != 0]
    pts2_filtered = pts_right[mask.ravel() != 0]

    # t3 = np.arange(retval)
    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    # axs[0, 0].plot(t3, pts1_filtered[:, 0], label='x1')
    # axs[0, 0].set_ylabel('x_leftcamara for calculate F')
    # axs[0, 1].plot(t3, pts1_filtered[:, 1], label='y1')
    # axs[0, 1].set_ylabel('y_leftcamera for calculate F')
    # axs[1, 0].plot(t3, pts2_filtered[:, 0], label='x2')
    # axs[1, 0].set_ylabel('x_rightcamara for calculate F')
    # axs[1, 1].plot(t3, pts2_filtered[:, 1], label='y2')
    # axs[1, 1].set_ylabel('y_rightcamara for calculate F')
    # plt.show()

    # Constructing the projection matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    # pts1 = np.hstack(pts1).T
    # pts2 = np.hstack(pts2).T
    X = cv2.triangulatePoints(P1, P2, pts1_filtered.T, pts2_filtered.T)
    # pts_3d = cv2.convertPointsFromHomogeneous(X.T)

    X = X[:3] / X[3]

    points_3d_homogeneous = np.hstack((X.T, np.ones((X.T.shape[0], 1)))).T
    P_W = np.dot(np.linalg.inv(P), points_3d_homogeneous)
    points_world = P_W[0:3, :]

    # Filter out points with large reprojection errors using RANSAC
    # X = X[:, mask.ravel() != 0]

    t2 = np.arange(len(X[0]))
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 9))
    axs[0].set_title('Positon_Reconstruction')
    axs[0].plot(t2, X[0, :].T, label='x')
    axs[0].set_ylabel('X')
    axs[1].plot(t2, X[1, :].T, label='y')
    axs[1].set_ylabel('Y')
    axs[2].plot(t2, X[2, :].T, label='Z')
    axs[2].set_ylabel('Z')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points_world[0], points_world[1], points_world[2], marker='.')
    ax.set_title('3d_Reconstruction')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    return points_world


def rtvec_to_matrix(rvec=(0, 0, 0), tvec=(0, 0, 0)):
    "Convert rotation vector and translation vector to 4x4 matrix"
    rvec = np.asarray(rvec)
    tvec = np.asarray(tvec)

    T = np.eye(4)
    (R, jac) = cv2.Rodrigues(rvec)
    T[:3, :3] = R
    T[:3, 3] = tvec.squeeze()
    return T


def read_sensor(df):

    current_df = df

    # left_arm = [12, 13, 14]
    # right_arm = [8, 9, 10]
    # left_leg = [19, 20, 21]
    # right_leg = [15, 16, 17]
    kp = []
    limbs = [12, 13, 14, 8, 9, 10, 19, 20, 21, 15, 16, 17]
    for i in range(0, 12):
        a = []
        for j in range(1, 4):
            a.append(current_df.iloc[:, limbs[i] * 3 + j].values.tolist())
        kp.append(a)
    return kp


def read_mp(df, zscore=True):
    joints = [11, 13, 15, 12, 14, 16, 23, 25, 27, 24, 26, 28]
    df = df
    current_df = df.drop('filename', axis=1)
    current_df = current_df.drop('timestamp [ms]', axis=1)
    if zscore == True:
        def zscore_scaler(x): return (x - np.min(x)) / (np.max(x) - np.min(x))
        current_df = current_df.apply(zscore_scaler)
    kp = []
    for i in range(0, 12):
        a = []
        for j in range(0, 3):
            a.append(current_df.iloc[:, joints[i] * 4 + j].values.tolist())
        kp.append(a)

    return kp


def video_img(path):
    cap = cv2.VideoCapture(path)

    # Initialize frame counter
    frame_count = 0

    # Loop through all frames
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()

        # Break the loop if we have reached the end of the video
        if not ret:
            break

        # Save the frame as a JPEG image
        cv2.imwrite(f'frame{frame_count:04d}.jpg', frame)

        # Increment the frame counter
        frame_count += 1

    # Release the video file
    cap.release()
    return


def angles_from_point_coordinates(x1, x2, x3):
    '''
    IN:
        x1: first point, either a 1D vector of [x,y(,z)] or a 2D array where the row elements are 1D vectors of [x,y(,z)]
        x2: middle point, of same shape as x1
        x3: third point, of same shape as x1
    RETURN:
        angle: either a scalar angle or a vector where each row element is an angle corresponding to the row values of the points
    '''
    first_leg = x2 - x1
    second_leg = x2 - x3
    if len(x1.shape) == 2:
        v1 = first_leg / np.linalg.norm(first_leg, axis=1)[:, None]
        v2 = second_leg / np.linalg.norm(second_leg, axis=1)[:, None]
        # angle = np.degrees(np.arccos(np.clip(np.sum(v1*v2,axis=1), -1.0, 1.0)))
        angle = np.degrees(np.arccos(np.sum(v1 * v2, axis=1)))
    else:
        v1 = first_leg / np.linalg.norm(first_leg)
        v2 = second_leg / np.linalg.norm(second_leg)
        angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
    return angle


def trajectory_length(trajectory_points):
    total_distance = 0
    for i in range(1, len(trajectory_points[0])):
        x1 = trajectory_points[0][i - 1]
        y1 = trajectory_points[1][i - 1]
        z1 = trajectory_points[2][i - 1]
        x2 = trajectory_points[0][i]
        y2 = trajectory_points[1][i]
        z2 = trajectory_points[2][i]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        total_distance += distance
    return total_distance


def trajectory_distance(P_s, P_c):
    dis = []
    for i in range(0, len(P_s[0])):
        x1 = P_s[0][i]
        y1 = P_s[1][i]
        z1 = P_s[2][i]
        x2 = P_c[0][i]
        y2 = P_c[1][i]
        z2 = P_c[2][i]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        dis.append(distance)
    d = np.array([dis])
    return d


def plot_sensor(path_sensor, start, end):
    # left_arm = [12, 13, 14]
    # right_arm = [8, 9, 10]
    # left_leg = [19, 20, 21]
    # right_leg = [15, 16, 17]
    kp = []
    limbs = [12, 13, 14, 8, 9, 10, 19, 20, 21, 15, 16, 17]

    df3 = pd.read_csv(path_sensor)
    df3 = df3[start:end]
    a = df3.iloc[:, (limbs[0]-1)*3+1]
    b = df3.iloc[:, (limbs[0]-1)*3+2]
    points = np.array([[df3.iloc[:, (limbs[0])*3+1], df3.iloc[:, (limbs[0])*3+2], df3.iloc[:, (limbs[0])*3+3]],
                       [df3.iloc[:, (limbs[3])*3+1], df3.iloc[:,
                                                              (limbs[3])*3+2], df3.iloc[:, (limbs[3])*3+3]],
                       [df3.iloc[:, (limbs[6])*3+1], df3.iloc[:,
                                                              (limbs[6])*3+2], df3.iloc[:, (limbs[6])*3+3]],
                       [df3.iloc[:, (limbs[9])*3+1], df3.iloc[:, (limbs[9])*3+2], df3.iloc[:, (limbs[9])*3+3]]])
    points2 = np.hstack(points)
    t = np.arange(len(points[0][0])) / 60.0  # Assuming 30 fps sample rate
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(12, 9))
    fig.suptitle('sensor points')
    axs[0, 0].plot(t, points[0][0], label='x')
    axs[0, 0].set_ylabel('left shoulder')
    axs[0, 0].set_title('X')
    axs[0, 1].set_title('Y')
    axs[0, 2].set_title('Z')
    axs[0, 0].legend()
    axs[0, 1].plot(t, points[0][1], label='y')
    axs[0, 2].plot(t, points[0][2], label='z')

    # Second row of subplots: x3 vs x4
    axs[1, 0].plot(t, points[1][0], label='x')
    axs[1, 0].set_ylabel('right shoulder')
    axs[1, 0].legend()
    axs[1, 1].plot(t, points[1][1], label='y')
    axs[1, 2].plot(t, points[1][2], label='z')

    # Third row of subplots: x5 vs x6
    axs[2, 0].plot(t, points[2][0], label='x')
    axs[2, 0].set_ylabel('left hip')
    axs[2, 0].legend()
    axs[2, 1].plot(t, points[2][1], label='y')
    axs[2, 2].plot(t, points[2][2], label='z')

    axs[3, 0].plot(t, points[3][0], label='x')
    axs[3, 0].set_ylabel('right hip')
    axs[3, 0].legend()
    axs[3, 1].plot(t, points[3][1], label='y')
    axs[3, 2].plot(t, points[3][2], label='z')

    # plt.savefig('sensor points')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points2[0], points2[1], points2[2], marker='.')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(-0.2, 0.8)
    ax.set_zlim(0.5, 1.5)
    ax.set_ylim(0, 1)
    # plt.savefig('3d_Sensor.jpg')
    plt.show()
    return


def pixel_to_world(camera_intrinsics, r, t, img_points):

    K_inv = camera_intrinsics.I
    R_inv = np.asmatrix(r).I
    R_inv_T = np.dot(R_inv, np.asmatrix(t))
    world_points = []
    coords = np.zeros((3, 1), dtype=np.float64)
    for img_point in img_points:
        coords[0] = img_point[0]
        coords[1] = img_point[1]
        coords[2] = 1.0
        cam_point = np.dot(K_inv, coords)
        cam_R_inv = np.dot(R_inv, cam_point)
        scale = R_inv_T[2][0] / cam_R_inv[2][0]
        scale_world = np.multiply(scale, cam_R_inv)
        world_point = np.asmatrix(scale_world) - np.asmatrix(R_inv_T)
        pt = np.zeros((3, 1), dtype=np.float64)
        pt[0] = world_point[0]
        pt[1] = world_point[1]
        pt[2] = 0
        world_points.append(pt.T.tolist())

    return world_points


def calculate_sim(index, pts_sen, pts_cam, path_camera, path_sensor, shift13, shift23, P_left, P_right):

    df = pd.read_csv(path_camera)
    video_name_list = df['filename'].unique()
    # [Biceps,Cor_Squat,Far_Squat,Rotation_lef,Slight_Squat]
    df1 = df.loc[df['filename'] == video_name_list[index]]
    df2 = df.loc[df['filename'] == video_name_list[index+1]]
    df3 = pd.read_csv(path_sensor)
    # cut off first
    df1, df2, df3 = cut_dataframe(df1, df2, df3, index)
    # df3 = df3[47200:48720]
    df1, df2, df3 = syn_dataframe(df1, df2, df3, shift13, shift23)
    joints = [11, 13, 15, 12, 14, 16, 23, 25, 27, 24, 26, 28]
    points_left = np.array([[df1.iloc[:, (joints[0])*4+2], df1.iloc[:, (joints[0])*4+3], df1.iloc[:, (joints[0])*4+4]],
                            [df1.iloc[:, (joints[1])*4+2], df1.iloc[:,
                                                                    (joints[1])*4+3], df1.iloc[:, (joints[1])*4+4]],
                            [df1.iloc[:, (joints[2])*4+2], df1.iloc[:,
                                                                    (joints[2])*4+3], df1.iloc[:, (joints[2])*4+4]],
                            [df1.iloc[:, (joints[3])*4+2], df1.iloc[:, (joints[3])
                                                                    * 4+3], df1.iloc[:, (joints[3])*4+4]],
                            [df1.iloc[:, (joints[4])*4+2], df1.iloc[:, (joints[4])
                                                                    * 4+3], df1.iloc[:, (joints[4])*4+4]],
                            [df1.iloc[:, (joints[5])*4+2], df1.iloc[:, (joints[5])
                                                                    * 4+3], df1.iloc[:, (joints[5])*4+4]],
                            [df1.iloc[:, (joints[6])*4+2], df1.iloc[:, (joints[6])
                                                                    * 4+3], df1.iloc[:, (joints[6])*4+4]],
                            [df1.iloc[:, (joints[7])*4+2], df1.iloc[:, (joints[7])
                                                                    * 4+3], df1.iloc[:, (joints[7])*4+4]],
                            [df1.iloc[:, (joints[8])*4+2], df1.iloc[:, (joints[8])
                                                                    * 4+3], df1.iloc[:, (joints[8])*4+4]],
                            [df1.iloc[:, (joints[9])*4+2], df1.iloc[:, (joints[9])
                                                                    * 4+3], df1.iloc[:, (joints[9])*4+4]],
                            [df1.iloc[:, (joints[10])*4+2], df1.iloc[:, (joints[10])
                                                                     * 4+3], df1.iloc[:, (joints[10])*4+4]],
                            [df1.iloc[:, (joints[11])*4+2], df1.iloc[:, (joints[11])*4+3], df1.iloc[:, (joints[11])*4+4]]
                            ])
    pts_left = np.hstack(points_left)
    points_right = np.array([[df2.iloc[:, (joints[0])*4+2], df2.iloc[:, (joints[0])*4+3], df2.iloc[:, (joints[0])*4+4]],
                             [df2.iloc[:, (joints[1])*4+2], df2.iloc[:,
                                                                     (joints[1])*4+3], df2.iloc[:, (joints[1])*4+4]],
                             [df2.iloc[:, (joints[2])*4+2], df2.iloc[:,
                                                                     (joints[2])*4+3], df2.iloc[:, (joints[2])*4+4]],
                             [df2.iloc[:, (joints[3])*4+2], df2.iloc[:, (joints[3])
                                                                     * 4+3], df2.iloc[:, (joints[3])*4+4]],
                             [df2.iloc[:, (joints[4])*4+2], df2.iloc[:, (joints[4])
                                                                     * 4+3], df2.iloc[:, (joints[4])*4+4]],
                             [df2.iloc[:, (joints[5])*4+2], df2.iloc[:, (joints[5])
                                                                     * 4+3], df2.iloc[:, (joints[5])*4+4]],
                             [df2.iloc[:, (joints[6])*4+2], df2.iloc[:, (joints[6])
                                                                     * 4+3], df2.iloc[:, (joints[6])*4+4]],
                             [df2.iloc[:, (joints[7])*4+2], df2.iloc[:, (joints[7])
                                                                     * 4+3], df2.iloc[:, (joints[7])*4+4]],
                             [df2.iloc[:, (joints[8])*4+2], df2.iloc[:, (joints[8])
                                                                     * 4+3], df2.iloc[:, (joints[8])*4+4]],
                             [df2.iloc[:, (joints[9])*4+2], df2.iloc[:, (joints[9])
                                                                     * 4+3], df2.iloc[:, (joints[9])*4+4]],
                             [df2.iloc[:, (joints[10])*4+2], df2.iloc[:, (joints[10])
                                                                      * 4+3], df2.iloc[:, (joints[10])*4+4]],
                             [df2.iloc[:, (joints[11])*4+2], df2.iloc[:, (joints[11])*4+3], df2.iloc[:, (joints[11])*4+4]]
                             ])
    pts_right = np.hstack(points_right)

    points_3d_homogeneous_left = np.hstack(
        (pts_left.T, np.ones((pts_left.T.shape[0], 1)))).T
    P_W_left = np.dot(np.linalg.inv(P_left), points_3d_homogeneous_left)
    points_world_left = P_W_left[0:3, :]

    points_3d_homogeneous_right = np.hstack(
        (pts_right.T, np.ones((pts_right.T.shape[0], 1)))).T
    P_W_right = np.dot(np.linalg.inv(P_right), points_3d_homogeneous_right)
    points_world_right = P_W_right[0:3, :]

    # resmaple
    # reconstruct points
    xc = np.linspace(0, len(pts_cam[0]), len(pts_cam[0]))
    y1c = pts_cam[0]
    y2c = pts_cam[1]
    y3c = pts_cam[2]
    f1c = interp1d(xc, y1c, kind='quadratic')
    f2c = interp1d(xc, y2c, kind='quadratic')
    f3c = interp1d(xc, y3c, kind='quadratic')

    xcnew = np.linspace(0, len(pts_cam[0]), len(pts_cam[0])*2)
    ycnew1 = f1c(xcnew)  # generate the y values for all x values in xnew
    ycnew2 = f2c(xcnew)
    ycnew3 = f3c(xcnew)
    pts_cam_resample = np.array([ycnew1, ycnew2, ycnew3])

    # left camera 3d points
    xl = np.linspace(0, len(pts_cam[0]), len(pts_cam[0]))
    y1l = points_world_left[0]
    y2l = points_world_left[1]
    y3l = points_world_left[2]
    f1l = interp1d(xl, y1l, kind='quadratic')
    f2l = interp1d(xl, y2l, kind='quadratic')
    f3l = interp1d(xl, y3l, kind='quadratic')

    xlnew = np.linspace(0, len(pts_cam[0]), len(pts_cam[0])*2)
    ylnew1 = f1l(xlnew)  # generate the y values for all x values in xnew
    ylnew2 = f2l(xlnew)
    ylnew3 = f3l(xlnew)
    points_world_left_resample = np.array([ylnew1, ylnew2, ylnew3])

    # right camera 3d points
    xr = np.linspace(0, len(pts_cam[0]), len(pts_cam[0]))
    y1r = points_world_right[0]
    y2r = points_world_right[1]
    y3r = points_world_right[2]
    f1r = interp1d(xr, y1r, kind='quadratic')
    f2r = interp1d(xr, y2r, kind='quadratic')
    f3r = interp1d(xr, y3r, kind='quadratic')

    xrnew = np.linspace(0, len(pts_cam[0]), len(pts_cam[0])*2)
    yrnew1 = f1r(xrnew)  # generate the y values for all x values in xnew
    yrnew2 = f2r(xrnew)
    yrnew3 = f3r(xrnew)
    points_world_right_resample = np.array([yrnew1, yrnew2, yrnew3])

    # zscore
    # points_world_left = zscore(points_world_left_resample, axis=1)
    # points_world_right = zscore(points_world_right_resample, axis=1)
    # pts_cam_resample_norm = zscore(pts_cam_resample, axis=1)
    # pts_sen_norm = zscore(pts_sen.T, axis=1)

    # 最大最小norm
    _range_s_0 = np.max(pts_sen.T[0])-np.min(pts_sen.T[0])
    _range_s_1 = np.max(pts_sen.T[1])-np.min(pts_sen.T[1])
    _range_s_2 = np.max(pts_sen.T[2])-np.min(pts_sen.T[2])
    pts_sen_x = (pts_sen.T[0]-np.min(pts_sen.T[0])) / _range_s_0
    pts_sen_y = (pts_sen.T[1]-np.min(pts_sen.T[1])) / _range_s_1
    pts_sen_z = (pts_sen.T[2]-np.min(pts_sen.T[2])) / _range_s_2
    pts_sen_norm = np.vstack([pts_sen_x, pts_sen_y, pts_sen_z])

    _range_c_0 = np.max(pts_cam_resample[0])-np.min(pts_cam_resample[0])
    _range_c_1 = np.max(pts_cam_resample[1])-np.min(pts_cam_resample[1])
    _range_c_2 = np.max(pts_cam_resample[2])-np.min(pts_cam_resample[2])
    pts_cam_x = (pts_cam_resample[0]-np.min(pts_cam_resample[0])) / _range_c_0
    pts_cam_y = (pts_cam_resample[1]-np.min(pts_cam_resample[1])) / _range_c_1
    pts_cam_z = (pts_cam_resample[2]-np.min(pts_cam_resample[2])) / _range_c_2
    pts_cam_resample_norm = np.vstack([pts_cam_x, pts_cam_y, pts_cam_z])

    _range_l_0 = np.max(
        points_world_left_resample[0])-np.min(points_world_left_resample[0])
    _range_l_1 = np.max(
        points_world_left_resample[1])-np.min(points_world_left_resample[1])
    _range_l_2 = np.max(
        points_world_left_resample[2])-np.min(points_world_left_resample[2])
    points_world_left_resample_x = (
        points_world_left_resample[0]-np.min(points_world_left_resample[0])) / _range_l_0
    points_world_left_resample_y = (
        points_world_left_resample[1]-np.min(points_world_left_resample[1])) / _range_l_1
    points_world_left_resample_z = (
        points_world_left_resample[2]-np.min(points_world_left_resample[2])) / _range_l_2
    points_world_left = np.vstack(
        [points_world_left_resample_x, points_world_left_resample_y, points_world_left_resample_z])

    _range_r_0 = np.max(
        points_world_right_resample[0])-np.min(points_world_right_resample[0])
    _range_r_1 = np.max(
        points_world_right_resample[1])-np.min(points_world_right_resample[1])
    _range_r_2 = np.max(
        points_world_right_resample[2])-np.min(points_world_right_resample[2])
    points_world_right_resample_x = (
        points_world_right_resample[0]-np.min(points_world_right_resample[0])) / _range_r_0
    points_world_right_resample_y = (
        points_world_right_resample[1]-np.min(points_world_right_resample[1])) / _range_r_1
    points_world_right_resample_z = (
        points_world_right_resample[2]-np.min(points_world_right_resample[2])) / _range_r_2
    points_world_right = np.vstack(
        [points_world_right_resample_x, points_world_right_resample_y, points_world_right_resample_z])

    # distance
    distances1 = np.linalg.norm(pts_sen_norm - pts_cam_resample_norm, axis=0)
    distances2 = np.linalg.norm(pts_sen_norm - points_world_left, axis=0)
    distances3 = np.linalg.norm(pts_sen_norm - points_world_right, axis=0)

    # ##visualization
    # ##3d plot
    fig = plt.figure()
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0],projection='3d')
    ax1.scatter(pts_sen_norm[0], pts_sen_norm[1], pts_sen_norm[2], marker='.')
    ax1.set_xlabel('X Label')
    ax1.set_ylabel('Y Label')
    ax1.set_zlabel('Z Label')

    ax2 = fig.add_subplot(gs[0, 1],projection='3d')
    ax2.scatter(pts_cam_resample_norm[0], pts_cam_resample_norm[1], pts_cam_resample_norm[2], marker='.')
    ax2.set_xlabel('X Label')
    ax2.set_ylabel('Y Label')
    ax2.set_zlabel('Z Label')

    ax3 = fig.add_subplot(gs[1, 0],projection='3d')
    ax3.scatter(points_world_left[0], points_world_left[1], points_world_left[2], marker='.')
    ax3.set_xlabel('X Label')
    ax3.set_ylabel('Y Label')
    ax3.set_zlabel('Z Label')

    ax4 = fig.add_subplot(gs[1, 1],projection='3d')
    ax4.scatter(points_world_right[0], points_world_right[1], points_world_right[2], marker='.')
    ax4.set_xlabel('X Label')
    ax4.set_ylabel('Y Label')
    ax4.set_zlabel('Z Label')
    plt.show()

    # distance
    fig1, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 9), sharey=True)
    t1 = np.arange(len(distances1.T)) / 60
    axs[0].plot(t1, distances1.T, label='error_reconstruction')
    axs[0].axhline(y=np.nanmean(distances1.T), color='red',
                   linestyle='--', linewidth=3, label='Avg')
    axs[0].legend()
    axs[1].plot(t1, distances2.T, label='error_camera_left')
    axs[1].axhline(y=np.nanmean(distances2.T), color='red',
                   linestyle='--', linewidth=3, label='Avg')
    axs[1].legend()
    axs[2].plot(t1, distances3.T, label='error_camera_right')
    axs[2].axhline(y=np.nanmean(distances3.T), color='red',
                   linestyle='--', linewidth=3, label='Avg')
    axs[2].legend()
    # plt.savefig('diff_before.jpg')
    plt.show()

    return pts_sen_norm, pts_cam_resample_norm, points_world_left, points_world_right


def compare_joints(index, pts_sen, pts_cam, path_camera, path_sensor, shift13, shift23, P_left, P_right):
    # 3d points from mediapipe
    df = pd.read_csv(path_camera)
    video_name_list = df['filename'].unique()
    # [Biceps,Cor_Squat,Far_Squat,Rotation_lef,Slight_Squat]
    # index2 df1[200:-150] df2[200:-150]
    # index8 df1 = df1[130:-300] df2 = df2[250:-140] df3 = df3[720:-438]
    df1 = df.loc[df['filename'] == video_name_list[index]]
    df2 = df.loc[df['filename'] == video_name_list[index+1]]
    df3 = pd.read_csv(path_sensor)
    # cut off first
    df1, df2, df3 = cut_dataframe(df1, df2, df3, index)
    # df3 = df3[47200:48720]

    df1, df2, df3 = syn_dataframe(df1, df2, df3, shift13, shift23)

    joints = [11, 13, 15, 12, 14, 16, 23, 25, 27, 24, 26, 28]
    NAME = ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "RIGHT_SHOULDER",
            "RIGHT_ELBOW", "RIGHT_WRIST", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE",
            "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"]
    points_left = np.array([[df1.iloc[:, (joints[0])*4+2], df1.iloc[:, (joints[0])*4+3], df1.iloc[:, (joints[0])*4+4]],
                            [df1.iloc[:, (joints[1])*4+2], df1.iloc[:,
                                                                    (joints[1])*4+3], df1.iloc[:, (joints[1])*4+4]],
                            [df1.iloc[:, (joints[2])*4+2], df1.iloc[:,
                                                                    (joints[2])*4+3], df1.iloc[:, (joints[2])*4+4]],
                            [df1.iloc[:, (joints[3])*4+2], df1.iloc[:, (joints[3])
                                                                    * 4+3], df1.iloc[:, (joints[3])*4+4]],
                            [df1.iloc[:, (joints[4])*4+2], df1.iloc[:, (joints[4])
                                                                    * 4+3], df1.iloc[:, (joints[4])*4+4]],
                            [df1.iloc[:, (joints[5])*4+2], df1.iloc[:, (joints[5])
                                                                    * 4+3], df1.iloc[:, (joints[5])*4+4]],
                            [df1.iloc[:, (joints[6])*4+2], df1.iloc[:, (joints[6])
                                                                    * 4+3], df1.iloc[:, (joints[6])*4+4]],
                            [df1.iloc[:, (joints[7])*4+2], df1.iloc[:, (joints[7])
                                                                    * 4+3], df1.iloc[:, (joints[7])*4+4]],
                            [df1.iloc[:, (joints[8])*4+2], df1.iloc[:, (joints[8])
                                                                    * 4+3], df1.iloc[:, (joints[8])*4+4]],
                            [df1.iloc[:, (joints[9])*4+2], df1.iloc[:, (joints[9])
                                                                    * 4+3], df1.iloc[:, (joints[9])*4+4]],
                            [df1.iloc[:, (joints[10])*4+2], df1.iloc[:, (joints[10])
                                                                     * 4+3], df1.iloc[:, (joints[10])*4+4]],
                            [df1.iloc[:, (joints[11])*4+2], df1.iloc[:, (joints[11])*4+3], df1.iloc[:, (joints[11])*4+4]]])
    pts_left = np.hstack(points_left)
    points_right = np.array([[df2.iloc[:, (joints[0])*4+2], df2.iloc[:, (joints[0])*4+3], df2.iloc[:, (joints[0])*4+4]],
                             [df2.iloc[:, (joints[1])*4+2], df2.iloc[:,
                                                                     (joints[1])*4+3], df2.iloc[:, (joints[1])*4+4]],
                             [df2.iloc[:, (joints[2])*4+2], df2.iloc[:,
                                                                     (joints[2])*4+3], df2.iloc[:, (joints[2])*4+4]],
                             [df2.iloc[:, (joints[3])*4+2], df2.iloc[:, (joints[3])
                                                                     * 4+3], df2.iloc[:, (joints[3])*4+4]],
                             [df2.iloc[:, (joints[4])*4+2], df2.iloc[:, (joints[4])
                                                                     * 4+3], df2.iloc[:, (joints[4])*4+4]],
                             [df2.iloc[:, (joints[5])*4+2], df2.iloc[:, (joints[5])
                                                                     * 4+3], df2.iloc[:, (joints[5])*4+4]],
                             [df2.iloc[:, (joints[6])*4+2], df2.iloc[:, (joints[6])
                                                                     * 4+3], df2.iloc[:, (joints[6])*4+4]],
                             [df2.iloc[:, (joints[7])*4+2], df2.iloc[:, (joints[7])
                                                                     * 4+3], df2.iloc[:, (joints[7])*4+4]],
                             [df2.iloc[:, (joints[8])*4+2], df2.iloc[:, (joints[8])
                                                                     * 4+3], df2.iloc[:, (joints[8])*4+4]],
                             [df2.iloc[:, (joints[9])*4+2], df2.iloc[:, (joints[9])
                                                                     * 4+3], df2.iloc[:, (joints[9])*4+4]],
                             [df2.iloc[:, (joints[10])*4+2], df2.iloc[:, (joints[10])
                                                                      * 4+3], df2.iloc[:, (joints[10])*4+4]],
                             [df2.iloc[:, (joints[11])*4+2], df2.iloc[:, (joints[11])*4+3], df2.iloc[:, (joints[11])*4+4]]])
    pts_right = np.hstack(points_right)

    points_3d_homogeneous_left = np.hstack(
        (pts_left.T, np.ones((pts_left.T.shape[0], 1)))).T
    P_W_left = np.dot(np.linalg.inv(P_left), points_3d_homogeneous_left)
    points_world_left = P_W_left[0:3, :]

    points_3d_homogeneous_right = np.hstack(
        (pts_right.T, np.ones((pts_right.T.shape[0], 1)))).T
    P_W_right = np.dot(np.linalg.inv(P_right), points_3d_homogeneous_right)
    points_world_right = P_W_right[0:3, :]

    # 3d points from reconstruction and sensor
    pts_sen = pts_sen.T
    len_sen = int(pts_sen.shape[1]/12)
    len_cam = int(pts_cam.shape[1]/12)
    joints_nummer = np.arange(0, 12)
    cam = []
    sen = []
    shift_list = []
    _range_s_0 = np.max(pts_sen[0])-np.min(pts_sen[0])
    _range_s_1 = np.max(pts_sen[1])-np.min(pts_sen[1])
    _range_s_2 = np.max(pts_sen[2])-np.min(pts_sen[2])

    _range_c_0 = np.max(pts_cam[0])-np.min(pts_cam[0])
    _range_c_1 = np.max(pts_cam[1])-np.min(pts_cam[1])
    _range_c_2 = np.max(pts_cam[2])-np.min(pts_cam[2])

    _range_l_0 = np.max(points_world_left[0])-np.min(points_world_left[0])
    _range_l_1 = np.max(points_world_left[1])-np.min(points_world_left[1])
    _range_l_2 = np.max(points_world_left[2])-np.min(points_world_left[2])

    _range_r_0 = np.max(points_world_right[0])-np.min(points_world_right[0])
    _range_r_1 = np.max(points_world_right[1])-np.min(points_world_right[1])
    _range_r_2 = np.max(points_world_right[2])-np.min(points_world_right[2])

    for i in joints_nummer:
       # tmp = np.hstack((np.array([current_df.iloc[splits[i]:splits[i+1]].iloc[:,joints*4+1]]).squeeze(),np.array([current_df.iloc[splits[i]:splits[i+1]].iloc[:,joints*4+2]]).squeeze()))
        tem_cam = pts_cam[:, i*len_cam:(i+1)*len_cam]
        tem_sen = pts_sen[:, i*len_sen:(i+1)*len_sen]
        tem_left = points_world_left[:, i*len_cam:(i+1)*len_cam]
        tem_right = points_world_right[:, i*len_cam:(i+1)*len_cam]

        xc = np.linspace(0, len_cam, len_cam)
        y1c = tem_cam[0]
        y2c = tem_cam[1]
        y3c = tem_cam[2]

        fc1 = interp1d(xc, y1c, kind='quadratic')
        fc2 = interp1d(xc, y2c, kind='quadratic')
        fc3 = interp1d(xc, y3c, kind='quadratic')
        # xnew contains all points at which you want to sample the interpolated function
        # linspace should go from your start time to your end time
        xcnew = np.linspace(0, len_cam, len_cam*2)
        ycnew1 = fc1(xcnew)  # generate the y values for all x values in xnew
        ycnew2 = fc2(xcnew)
        ycnew3 = fc3(xcnew)
        tem_cam_resample = np.array([ycnew1, ycnew2, ycnew3])

        xl = np.linspace(0, len_cam, len_cam)
        y1l = tem_left[0]
        y2l = tem_left[1]
        y3l = tem_left[2]

        fl1 = interp1d(xl, y1l, kind='quadratic')
        fl2 = interp1d(xl, y2l, kind='quadratic')
        fl3 = interp1d(xl, y3l, kind='quadratic')
        # xnew contains all points at which you want to sample the interpolated function
        # linspace should go from your start time to your end time
        xlnew = np.linspace(0, len_cam, len_cam*2)
        ylnew1 = fl1(xlnew)  # generate the y values for all x values in xnew
        ylnew2 = fl2(xlnew)
        ylnew3 = fl3(xlnew)
        tem_left_resample = np.array([ylnew1, ylnew2, ylnew3])

        xr = np.linspace(0, len_cam, len_cam)
        y1r = tem_right[0]
        y2r = tem_right[1]
        y3r = tem_right[2]

        fr1 = interp1d(xr, y1r, kind='quadratic')
        fr2 = interp1d(xr, y2r, kind='quadratic')
        fr3 = interp1d(xr, y3r, kind='quadratic')
        # xnew contains all points at which you want to sample the interpolated function
        # linspace should go from your start time to your end time
        xrnew = np.linspace(0, len_cam, len_cam*2)
        yrnew1 = fr1(xrnew)  # generate the y values for all x values in xnew
        yrnew2 = fr2(xrnew)
        yrnew3 = fr3(xrnew)
        tem_right_resample = np.array([yrnew1, yrnew2, yrnew3])

        # _range_s_0= np.max(tem_sen[0])-np.min(tem_sen[0])
        # _range_s_1= np.max(tem_sen[1])-np.min(tem_sen[1])
        # _range_s_2= np.max(tem_sen[2])-np.min(tem_sen[2])
        pts_sen_x = (tem_sen[0]-np.min(tem_sen[0])) / _range_s_0
        pts_sen_y = (tem_sen[1]-np.min(tem_sen[1])) / _range_s_1
        pts_sen_z = (tem_sen[2]-np.min(tem_sen[2])) / _range_s_2
        P_s = np.vstack([pts_sen_x, pts_sen_y, pts_sen_z])

        # _range_c_0= np.max(tem_cam_resample[0])-np.min(tem_cam_resample[0])
        # _range_c_1= np.max(tem_cam_resample[1])-np.min(tem_cam_resample[1])
        # _range_c_2= np.max(tem_cam_resample[2])-np.min(tem_cam_resample[2])
        pts_cam_x = (tem_cam_resample[0] -
                     np.min(tem_cam_resample[0])) / _range_c_0
        pts_cam_y = (tem_cam_resample[1] -
                     np.min(tem_cam_resample[1])) / _range_c_1
        pts_cam_z = (tem_cam_resample[2] -
                     np.min(tem_cam_resample[2])) / _range_c_2
        P_c = np.vstack([pts_cam_x, pts_cam_y, pts_cam_z])

        pts_left_x = (tem_left_resample[0] -
                      np.min(tem_left_resample[0])) / _range_l_0
        pts_left_y = (tem_left_resample[1] -
                      np.min(tem_left_resample[1])) / _range_l_1
        pts_left_z = (tem_left_resample[2] -
                      np.min(tem_left_resample[2])) / _range_l_2
        P_l = np.vstack([pts_left_x, pts_left_y, pts_left_z])

        pts_right_x = (
            tem_right_resample[0]-np.min(tem_right_resample[0])) / _range_r_0
        pts_right_y = (
            tem_right_resample[1]-np.min(tem_right_resample[1])) / _range_r_1
        pts_right_z = (
            tem_right_resample[2]-np.min(tem_right_resample[2])) / _range_r_2
        P_r = np.vstack([pts_right_x, pts_right_y, pts_right_z])

        distances1 = np.linalg.norm(P_s - P_c, axis=0)
        distances2 = np.linalg.norm(P_s - P_l, axis=0)
        distances3 = np.linalg.norm(P_s - P_r, axis=0)
        # a = P_s.T
        # b = P_c.T
        # corr = correlate((a[0:len(a),2]-np.min(a[0:len(a),2]))/(np.max(a[0:len(a),2])-np.min(a[0:len(a),2])), (b[0:len(b),2]-np.min(b[0:len(b),2]))/(np.max(b[0:len(b),2])-np.min(b[0:len(b),2])), mode='full')
        # shift = np.argmax(corr) - (len(b[0:len(b),2]) - 1)
        # shift_list.append(shift)
        # if shift >= 0:
        #     a = a[shift:]
        #     b = b
        # else:
        #     a = a
        #     b = b[-shift:]
        # length = min(len(a), len(b))

        # a = a[:length]
        # b = b[:length]
        # distances = np.linalg.norm(a.T - b.T, axis=0)

        # P_s = a.T
        # P_c = b.T

        cam.append(P_c)
        sen.append(P_s)
        # # plot

        # error
        fig1, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 9),sharey=True)
        t1 = np.arange(len(distances1.T))/60
        axs[0].plot(t1, distances1.T, label='error_reconstruction')
        axs[0].axhline(y=np.nanmean(distances1.T), color='red',
                       linestyle='--', linewidth=3, label='Avg')
        axs[0].set_title('Error from joint {}'.format(NAME[i]))
        axs[0].legend()
        axs[1].plot(t1, distances2.T, label='error_camera_left')
        axs[1].axhline(y=np.nanmean(distances2.T), color='red',
                       linestyle='--', linewidth=3, label='Avg')
        axs[1].legend()
        axs[2].plot(t1, distances3.T, label='error_camera_right')
        axs[2].axhline(y=np.nanmean(distances3.T), color='red',
                       linestyle='--', linewidth=3, label='Avg')
        axs[2].legend()
        plt.show()

        # x,y,z compare
        fig2, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 9))
        t1 = np.arange(len(distances1.T))
        axs[0].plot(t1, P_c[0], label='x_Reconstruction')
        axs[0].plot(t1, P_l[0], label='x_camera_left')
        axs[0].plot(t1, P_r[0], label='x_camera_right')
        axs[0].plot(t1, P_s[0], label='x_sensor')
        axs[0].set_title('Positon {}'.format(NAME[i]))
        axs[0].legend()
        axs[1].plot(t1, P_c[1], label='y_Reconstruction')
        axs[1].plot(t1, P_l[1], label='y_camera_left')
        axs[1].plot(t1, P_r[1], label='y_camera_right')
        axs[1].plot(t1, P_s[1], label='y_sensor')
        axs[1].legend()
        axs[2].plot(t1, P_c[2], label='y_Reconstruction')
        axs[2].plot(t1, P_l[2], label='y_camera_left')
        axs[2].plot(t1, P_r[2], label='y_camera_right')
        axs[2].plot(t1, P_s[2], label='y_sensor')
        axs[2].legend()
        plt.show()

        # # 3d plot

        fig = plt.figure(figsize=plt.figaspect(0.5))
        t1 = np.arange(len(distances1.T))
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        ax1.scatter(P_c[0], P_c[1], P_c[2], marker='.')
        ax1.set_xlabel('X Label')
        ax1.set_ylabel('Y Label')
        ax1.set_zlabel('Z Label')
        ax1.set_title('reconstruction')
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        ax2.scatter(P_s[0], P_s[1], P_s[2], marker='.')
        ax2.set_xlabel('X Label')
        ax2.set_ylabel('Y Label')
        ax2.set_zlabel('Z Label')
        ax2.set_title('Sensor')

        ax3 = fig.add_subplot(gs[1, 0], projection='3d')
        ax3.scatter(P_l[0], P_l[1], P_l[2], marker='.')
        ax3.set_xlabel('X Label')
        ax3.set_ylabel('Y Label')
        ax3.set_zlabel('Z Label')
        ax3.set_title('camera_left')

        ax4 = fig.add_subplot(gs[1, 1], projection='3d')
        ax4.scatter(P_r[0], P_r[1], P_r[2], marker='.')
        ax4.set_xlabel('X Label')
        ax4.set_ylabel('Y Label')
        ax4.set_zlabel('Z Label')
        ax4.set_title('camera_right')
        plt.show()

    return shift_list


def compare_joints_worlds(index, path_camera, path_sensor, shift13, shift23, P_left, P_right):
    size = np.array([2394, 1684])
    df = pd.read_csv(path_camera)
    video_name_list = df['filename'].unique()
    # [Biceps,Cor_Squat,Far_Squat,Rotation_lef,Slight_Squat]
    # index2 df1[200:-150] df2[200:-150]
    # index8 df1 = df1[130:-300] df2 = df2[250:-140] df3 = df3[720:-438]
    df1 = df.loc[df['filename'] == video_name_list[index]]
    df2 = df.loc[df['filename'] == video_name_list[index+1]]
    df3 = pd.read_csv(path_sensor)
    # cut off first
    df1, df2, df3 = cut_dataframe(df1, df2, df3, index)
    # df3 = df3[47200:48720]

    df1, df2, df3 = syn_dataframe(df1, df2, df3, shift13, shift23)

    joints = [11, 13, 15, 12, 14, 16, 23, 25, 27, 24, 26, 28]
    points_left = np.array([[df1.iloc[:, (joints[0])*4+2], df1.iloc[:, (joints[0])*4+3], df1.iloc[:, (joints[0])*4+4]],
                            [df1.iloc[:, (joints[1])*4+2], df1.iloc[:,
                                                                    (joints[1])*4+3], df1.iloc[:, (joints[1])*4+4]],
                            [df1.iloc[:, (joints[2])*4+2], df1.iloc[:,
                                                                    (joints[2])*4+3], df1.iloc[:, (joints[2])*4+4]],
                            [df1.iloc[:, (joints[3])*4+2], df1.iloc[:, (joints[3])
                                                                    * 4+3], df1.iloc[:, (joints[3])*4+4]],
                            [df1.iloc[:, (joints[4])*4+2], df1.iloc[:, (joints[4])
                                                                    * 4+3], df1.iloc[:, (joints[4])*4+4]],
                            [df1.iloc[:, (joints[5])*4+2], df1.iloc[:, (joints[5])
                                                                    * 4+3], df1.iloc[:, (joints[5])*4+4]],
                            [df1.iloc[:, (joints[6])*4+2], df1.iloc[:, (joints[6])
                                                                    * 4+3], df1.iloc[:, (joints[6])*4+4]],
                            [df1.iloc[:, (joints[7])*4+2], df1.iloc[:, (joints[7])
                                                                    * 4+3], df1.iloc[:, (joints[7])*4+4]],
                            [df1.iloc[:, (joints[8])*4+2], df1.iloc[:, (joints[8])
                                                                    * 4+3], df1.iloc[:, (joints[8])*4+4]],
                            [df1.iloc[:, (joints[9])*4+2], df1.iloc[:, (joints[9])
                                                                    * 4+3], df1.iloc[:, (joints[9])*4+4]],
                            [df1.iloc[:, (joints[10])*4+2], df1.iloc[:, (joints[10])
                                                                     * 4+3], df1.iloc[:, (joints[10])*4+4]],
                            [df1.iloc[:, (joints[11])*4+2], df1.iloc[:, (joints[11])*4+3], df1.iloc[:, (joints[11])*4+4]]])
    pts_left = np.hstack(points_left)
    points_right = np.array([[df2.iloc[:, (joints[0])*4+2], df2.iloc[:, (joints[0])*4+3], df2.iloc[:, (joints[0])*4+4]],
                             [df2.iloc[:, (joints[1])*4+2], df2.iloc[:,
                                                                     (joints[1])*4+3], df2.iloc[:, (joints[1])*4+4]],
                             [df2.iloc[:, (joints[2])*4+2], df2.iloc[:,
                                                                     (joints[2])*4+3], df2.iloc[:, (joints[2])*4+4]],
                             [df2.iloc[:, (joints[3])*4+2], df2.iloc[:, (joints[3])
                                                                     * 4+3], df2.iloc[:, (joints[3])*4+4]],
                             [df2.iloc[:, (joints[4])*4+2], df2.iloc[:, (joints[4])
                                                                     * 4+3], df2.iloc[:, (joints[4])*4+4]],
                             [df2.iloc[:, (joints[5])*4+2], df2.iloc[:, (joints[5])
                                                                     * 4+3], df2.iloc[:, (joints[5])*4+4]],
                             [df2.iloc[:, (joints[6])*4+2], df2.iloc[:, (joints[6])
                                                                     * 4+3], df2.iloc[:, (joints[6])*4+4]],
                             [df2.iloc[:, (joints[7])*4+2], df2.iloc[:, (joints[7])
                                                                     * 4+3], df2.iloc[:, (joints[7])*4+4]],
                             [df2.iloc[:, (joints[8])*4+2], df2.iloc[:, (joints[8])
                                                                     * 4+3], df2.iloc[:, (joints[8])*4+4]],
                             [df2.iloc[:, (joints[9])*4+2], df2.iloc[:, (joints[9])
                                                                     * 4+3], df2.iloc[:, (joints[9])*4+4]],
                             [df2.iloc[:, (joints[10])*4+2], df2.iloc[:, (joints[10])
                                                                      * 4+3], df2.iloc[:, (joints[10])*4+4]],
                             [df2.iloc[:, (joints[11])*4+2], df2.iloc[:, (joints[11])*4+3], df2.iloc[:, (joints[11])*4+4]]])
    pts_right = np.hstack(points_right)

    points_3d_homogeneous_left = np.hstack(
        (pts_left.T, np.ones((pts_left.T.shape[0], 1)))).T
    P_W_left = np.dot(np.linalg.inv(P_left), points_3d_homogeneous_left)
    points_world_left = P_W_left[0:3, :]

    points_3d_homogeneous_right = np.hstack(
        (pts_right.T, np.ones((pts_right.T.shape[0], 1)))).T
    P_W_right = np.dot(np.linalg.inv(P_right), points_3d_homogeneous_right)
    points_world_right = P_W_right[0:3, :]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points_world_right[0], points_world_right[1],
               points_world_right[2], marker='.')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # t1 = np.arange(len(pts_right))
    # gs = GridSpec(2, 2, figure=fig)
    # ax1 = fig.add_subplot(gs[0,0], projection='3d')
    # ax1.scatter(pts_left[0], pts_left[1], pts_left[2], marker='.')
    # ax1.set_xlabel('X Label')
    # ax1.set_ylabel('Y Label')
    # ax1.set_zlabel('Z Label')
    # ax2 = fig.add_subplot(gs[0,1], projection='3d')
    # ax2.scatter(pts_right[0], pts_right[1], pts_right[2], marker='.')
    # ax2.set_xlabel('X Label')
    # ax2.set_ylabel('Y Label')
    # ax2.set_zlabel('Z Label')
    # ax3 = fig.add_subplot(gs[1,0], projection='3d')
    # ax3.scatter(points_world_left[0], points_world_left[1], points_world_left[2], marker='.')
    # ax3.set_xlabel('X Label')
    # ax3.set_ylabel('Y Label')
    # ax3.set_zlabel('Z Label')
    # ax4 = fig.add_subplot(gs[1,1], projection='3d')
    # ax4.scatter(points_world_right[0], points_world_right[1], points_world_right[2], marker='.')
    # ax4.set_xlabel('X Label')
    # ax4.set_ylabel('Y Label')
    # ax4.set_zlabel('Z Label')
    plt.show()
    return df1, df2, df3


def syn_dataframe(df1, df2, df3, shift13, shift23):
    if shift13 >= 0:
        if shift23 >= 0:
            if shift13 >= shift23:
                df3 = df3[shift13:]
                df2 = df2[round((shift13 - shift23)/2):]
                df1 = df1[:min(len(df2), round(len(df3)/2)
                               )]
            else:
                df3 = df3[shift23:]
                df1 = df1[round((shift23 - shift13)/2):]
                df2 = df2[:min(len(df1), round(len(df3)/2)
                               )]
        elif shift23 < 0:
            df3 = df3[shift13:]
            df2 = df2[round((shift13 - shift23)/2):]
            df1 = df1[:min(len(df2), round(len(df3)/2))]
    elif shift13 < 0:
        if shift23 >= 0:
            df3 = df3[shift23:]
            df1 = df1[round((shift23 - shift13)/2):]
            df2 = df2[:min(len(df1), round(len(df3)/2))]
        elif shift23 < 0:
            if shift13 >= shift23:
                df2 = df2[-round(shift23/2):]
                df1 = df1[-round(shift13/2):]
                df3 = df3[:min(len(df1)*2, len(df2)*2
                               )]
            else:
                df2 = df2[-round(shift23/2):]
                df1 = df1[-round(shift13/2):]
                df3 = df3[:min(len(df1)*2, len(df2)*2
                               )]
    length = min(len(df1), len(df2), round(len(df3)/2))

    df1 = df1[:length]
    df2 = df2[:length]
    df3 = df3[:length*2]
    return df1, df2, df3


def cut_dataframe(df1, df2, df3, index):
    # [Biceps,Cor_Squat,Far_Squat,Rotation_lef,Slight_Squat]
    if index == 0:
        df1 = df1[217:-282]
        df2 = df2[313:-97]
        df3 = df3[900:-374]
    elif index == 2:
        df1 = df1[300:-280]
        df2 = df2[200:-150]
        df3 = df3[864:-504]
    elif index == 4:
        df1 = df1[200:-378]
        df2 = df2[283:-115]
        df3 = df3[796:-389]
    elif index == 8:
        df1 = df1[130:-300]
        df2 = df2[250:-140]
        df3 = df3[720:-438]
    return df1, df2, df3


mtx = np.array([[2.02354675e+03, 0.00000000e+00, 1.33796374e+03],
                [0.00000000e+00, 2.02505711e+03, 9.80799093e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array(
    [-0.30198934, -0.01203354, -0.00079829, -0.00049461, 0.07992867])
P11 = np.array([[-0.37144907,  0.9272702,  0.04685677, -0.31930919],
                [-0.23162339, -0.0436763, -0.97182456,  1.18659086],
                [-0.89909743, -0.37183645,  0.23100099,  0.39905917],
                [0.,  0.,  0.,  1.]])
P22 = np.array([[-0.12835448,  0.99144406,  0.02374452, -1.50651084],
                [0.02006785,  0.0265342, -0.99944646,  0.61018696],
                [-0.99152529, -0.12780693, -0.02330194,  3.81251655],
                [0.,  0.,  0.,  1.]])
# mp_path = 'mediapipe_output.csv'
# leftpoints =read_mp('pos_left.csv')
# rightpoints =read_mp('pos_right.csv')
# print(leftpoints)
path_sensor = [r"F:\studium\Masterarbeit\X\Data\05302023_X_Biceps_sensor_Segment Position.csv",
               r"F:\studium\Masterarbeit\X\Data\05302023_X_Cor_Squat_sensor_Segment Position.csv",
               r"F:\studium\Masterarbeit\X\Data\05302023_X_Far_Squat_sensor_Segment Position.csv",
               r"F:\studium\Masterarbeit\X\Data\05302023_X_Rotation_sensor_Segment Position.csv",
               r"F:\studium\Masterarbeit\X\Data\05302023_X_Slight_Squat_sensor_Segment Position.csv"]
path_camera_landmarks = r"F:\studium\Masterarbeit\X\Data\X_data_undist_landmarks.csv"
path_camera_worldmarks = r"F:\studium\Masterarbeit\X\Data\X_data_undist_worldmarks.csv"


# path_sensor = "F:\\studium\\Masterarbeit\\projectcode\\Segment Position.csv"
# path_camera_landmarks = "F:\\studium\\Masterarbeit\\videos\\pos_left_right_undist.csv"
# path_camera_worldmarks = "F:\\studium\\Masterarbeit\\videos\\data_undist_worldmarks.csv"
#


index_joint_left = 2
shift13, shift23, shift12 = frame_diff(index_joint_left,
                                       path_camera_landmarks, path_sensor[int(index_joint_left/2)], worldmarks=False, save=False)

pts_3d, P_left, P_right, rl, tl, rr, tr = rot_trans_sen(index_joint_left,
                                                        path_sensor[int(index_joint_left/2)], path_camera_landmarks, mtx, dist, shift13, shift23)

points_world = rot_trans_cam(index_joint_left,
                             path_camera_landmarks, path_sensor[int(index_joint_left/2)], mtx, dist, P_left, shift13, shift23)

cor_sen, cor_rec, cor_left, cor_right = calculate_sim(index_joint_left,
                                                      pts_3d, points_world, path_camera_worldmarks, path_sensor[int(index_joint_left/2)], shift13, shift23, P_left, P_right)


# x = np.arange(len(cor_sen[2])/12)
# plt.plot(x,cor_rec[2][:int(len(cor_sen[2])/12)])
# plt.show()


# # print(distance)
shift = compare_joints(index_joint_left, pts_3d, points_world,
                       path_camera_worldmarks, path_sensor[int(index_joint_left/2)], shift13, shift23, P_left, P_right)
# # print(shift)
# # shift13,shift23,shift12 = frame_diff(path_camera_worldmarks,path_sensor, 47200, 48720, worldmarks = True,save=False)
# a = compare_joints_worlds(index_joint_left, path_camera_worldmarks,
#                           path_camera_worldmarks, shift13, shift23, P_left, P_right)
# # plot_sensor('Segment Position.csv', 47200, 48720)
# df = pd.DataFrame(X.T)
# column_names = ['x','y','z']
# output_df = df.set_axis(column_names,axis=1)
# output_df.to_csv('reconstructed_3D_points_8_with_filter.csv', index=False)
# 投影到3d对比
# 传感器的点
# sensor_data_path = 'Segment Position.csv'
# points_sensor = read_sensor('Segment Position.csv')
# mtx_l, dist_l = camera_calibration(path[0])
# mtx_r, dist_r = camera_calibration(path[1])
# path = ["F:\\studium\\Masterarbeit\\Xin_Recording_Cam\\05302023_P_Slight_Squat_left.mp4",
# "F:\\studium\\Masterarbeit\\Xin_Recording_Cam\\05302023_P_Far_Squat_right.mp4",
# "F:\\studium\\Masterarbeit\\Xin_Recording_Cam\\05302023_P_Far_Squat_left.mp4",
# "F:\\studium\\Masterarbeit\\Xin_Recording_Cam\\05302023_P_Biceps_right.mp4",
# "F:\\studium\\Masterarbeit\\Xin_Recording_Cam\\05302023_P_Biceps_left.mp4",
# "F:\\studium\\Masterarbeit\\Xin_Recording_Cam\\05302023_P_Rotation_right.mp4",
# "F:\\studium\\Masterarbeit\\Xin_Recording_Cam\\05302023_P_Rotation_left.mp4",
# "F:\\studium\\Masterarbeit\\Xin_Recording_Cam\\05302023_P_Cor_Squat_right.mp4",
# "F:\\studium\\Masterarbeit\\Xin_Recording_Cam\\05302023_P_Cor_Squat_left.mp4",
# "F:\\studium\\Masterarbeit\\Xin_Recording_Cam\\05302023_P_Slight_Squat_right.mp4"]
# for i in path:
#     undistortpath_ = undistort_video(i, mtx, dist, show=False, save=True, start_frame=0, end_frame=32000)
# # 报错，路径多加了一组引号
# undistortpath_r = undistort_video(path[1], mtx, dist, show=False, save=True, start_frame=1, end_frame=32000)
# undistortpath = [undistortpath_l,undistortpath_r]
# print(undistortpath)
# # #
# F= F_matrix(undistortpath)
#
# # Loop through each sheet and save as csv

# xlsx_file = pd.read_excel(r"F:\studium\Masterarbeit\X\Data\05302023_X_Rotation_sensor.xlsx", sheet_name=None)
# for sheet_name, sheet_data in xlsx_file.items():
#     sheet_data.to_csv(f'{sheet_name}.csv', index=False)
# df = pd.read_excel(sensor_data_path,sheet_name=1)
# keypoints_sens_3D = fg.read_csv(sensor_data_path)
