import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import svd
from scipy.signal import correlate, resample_poly




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

# 去除失真得到视频
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

# 计算R,t,P,投影得到3d点

def rot_trans_cam(path, K, dist,P):

    df1 = pd.read_csv(path[0])
    df2 = pd.read_csv(path[1])
    leftpoints = read_mp(df1, zscore=False)
    rightpoints = read_mp(df2, zscore=False)

    size = np.array([2394, 1684])
    # size = np.array([2560, 1920])
    pts1 = np.hstack(
        np.array([[leftpoints[0][0], leftpoints[0][1]],
                  [leftpoints[1][0], leftpoints[1][1]],
                  [leftpoints[2][0], leftpoints[2][1]],
                  [leftpoints[3][0], leftpoints[3][1]],
                  [leftpoints[4][0], leftpoints[4][1]],
                  [leftpoints[5][0], leftpoints[5][1]],
                  [leftpoints[6][0], leftpoints[6][1]],
                  [leftpoints[7][0], leftpoints[7][1]],
                  [leftpoints[8][0], leftpoints[8][1]],
                  [leftpoints[9][0], leftpoints[9][1]],
                  [leftpoints[10][0], leftpoints[10][1]],
                  [leftpoints[11][0], leftpoints[11][1]]])).T * size
    pts2 = np.hstack(
        np.array([[rightpoints[0][0], rightpoints[0][1]],
                  [rightpoints[1][0], rightpoints[1][1]],
                  [rightpoints[2][0], rightpoints[2][1]],
                  [rightpoints[3][0], rightpoints[3][1]],
                  [rightpoints[4][0], rightpoints[4][1]],
                  [rightpoints[5][0], rightpoints[5][1]],
                  [rightpoints[6][0], rightpoints[6][1]],
                  [rightpoints[7][0], rightpoints[7][1]],
                  [rightpoints[8][0], rightpoints[8][1]],
                  [rightpoints[9][0], rightpoints[9][1]],
                  [rightpoints[10][0], rightpoints[10][1]],
                  [rightpoints[11][0], rightpoints[11][1]]])).T * size

    # pts11 = pts1 * size
    # pts22 = pts2 * size
    # calculate inverse
    # my_inverse = np.linalg.inv(np.array([leftpoints[0][0]]))
    # Intrinsic matrix and distortion coefficients
    # K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # dist_coef = np.array([k1, k2, p1, p2, k3])
    K = K*2
    dist_coef = dist

    # Normalize image coordinates of corresponding points
    # pts1_norm = cv2.undistortPoints(pts1, K, dist_coef)
    # pts1_norm = pts1_norm.reshape(-1, 2)
    # pts2_norm = cv2.undistortPoints(pts2, K, dist_coef)
    # pts2_norm = pts2_norm.reshape(-1, 2)
    m = len(pts1.T[0])
    n = len(pts1[:, 0])
    t1 = np.arange(len(pts1.T[0]))
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    axs[0, 0].plot(t1, pts1[:, 0], label='x1')
    axs[0, 0].set_ylabel('left shoulder')
    axs[0, 1].plot(t1, pts1[:, 1], label='y1')
    axs[0, 1].set_ylabel('right shoulder')
    axs[1, 0].plot(t1, pts2[:, 0], label='x2')
    axs[1, 0].set_ylabel('left hip')
    axs[1, 1].plot(t1, pts2[:, 1], label='y2')
    axs[1, 1].set_ylabel('right hip')

    plt.show()
    # Compute fundamental matrix from normalized points
    # F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    F, mask1 = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC)
    # F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    # Compute essential matrix from fundamental matrix and intrinsic matrix
    E = K.T @ F @ K

    # p1, p2 = cv2.correctMatches(E, pts1.T, pts2.T)

    retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    mask = mask.astype('uint8')
    pts1_filtered = pts1[mask.ravel() != 0]
    pts2_filtered = pts2[mask.ravel() != 0]

    t3 = np.arange(retval)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    axs[0, 0].plot(t3, pts1_filtered[:, 0], label='x1')
    axs[0, 0].set_ylabel('left shoulder')
    axs[0, 1].plot(t3, pts1_filtered[:, 1], label='y1')
    axs[0, 1].set_ylabel('right shoulder')
    axs[1, 0].plot(t3, pts2_filtered[:, 0], label='x2')
    axs[1, 0].set_ylabel('left hip')
    axs[1, 1].plot(t3, pts2_filtered[:, 1], label='y2')
    axs[1, 1].set_ylabel('right hip')
    plt.show()

    # Constructing the projection matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    # pts1 = np.hstack(pts1).T
    # pts2 = np.hstack(pts2).T
    X = cv2.triangulatePoints(P1, P2, pts1_filtered.T, pts2_filtered.T)
    # pts_3d = cv2.convertPointsFromHomogeneous(X.T)

    X = X[:3] / X[3]

    points_3d_homogeneous= np.hstack((X.T, np.ones((X.T.shape[0], 1)))).T
    P_W = np.dot(np.linalg.inv(P),points_3d_homogeneous)
    points_world = P_W[0:3,:]
    
    # Filter out points with large reprojection errors using RANSAC
    # X = X[:, mask.ravel() != 0]
    t2 = np.arange(len(X[0]))
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 9))
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


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # ax.set_xlim(-0.2, 0.8)
    # ax.set_zlim(0.5, 1.5)
    # ax.set_ylim(0, 1)
    plt.savefig('3d_reconstruction_FM_8POINT_all.jpg')
    plt.show()
    return points_world, F, E, R, t, P1, P2, X

def rot_trans_sen(path_sensor, path_camera, cam, dist, shift13):

    size = np.array([2394, 1684])
    df1 = pd.read_csv(path_sensor)
    df1 = df1[47200:48720]
    df2 = pd.read_csv(path_camera)

    if shift13 >= 0:
        df1 = df1[shift13:]
        length = min(len(df1)/2, len(df2))
        df2 = df2[:length]
        df1 = df1[:length*2]

    # left_arm = [12, 13, 14]
    # right_arm = [8, 9, 10]
    # left_leg = [19, 20, 21]
    # right_leg = [15, 16, 17]
    kp = []
    limbs = [12, 13, 14, 8, 9, 10, 19, 20, 21, 15, 16, 17]

    # points = np.array([[df1.iloc[:, (limbs[0])*3+1], df1.iloc[:, (limbs[0])*3+2], df1.iloc[:, (limbs[0])*3+3]],
    #                    [df1.iloc[:, (limbs[1])*3+1], df1.iloc[:, (limbs[1])*3+2], df1.iloc[:, (limbs[1])*3+3]],
    #                    [df1.iloc[:, (limbs[2])*3+1], df1.iloc[:, (limbs[2])*3+2], df1.iloc[:, (limbs[2])*3+3]],
    #                    [df1.iloc[:, (limbs[3])*3+1], df1.iloc[:, (limbs[3])*3+2], df1.iloc[:, (limbs[3])*3+3]],
    #                    [df1.iloc[:, (limbs[4])*3+1], df1.iloc[:, (limbs[4])*3+2], df1.iloc[:, (limbs[4])*3+3]],
    #                    [df1.iloc[:, (limbs[5])*3+1], df1.iloc[:, (limbs[5])*3+2], df1.iloc[:, (limbs[5])*3+3]],
    #                    [df1.iloc[:, (limbs[6])*3+1], df1.iloc[:, (limbs[6])*3+2], df1.iloc[:, (limbs[6])*3+3]],
    #                    [df1.iloc[:, (limbs[7])*3+1], df1.iloc[:, (limbs[7])*3+2], df1.iloc[:, (limbs[7])*3+3]],
    #                    [df1.iloc[:, (limbs[8])*3+1], df1.iloc[:, (limbs[8])*3+2], df1.iloc[:, (limbs[8])*3+3]],
    #                    [df1.iloc[:, (limbs[9])*3+1], df1.iloc[:, (limbs[9])*3+2], df1.iloc[:, (limbs[9])*3+3]],
    #                    [df1.iloc[:, (limbs[10])*3+1], df1.iloc[:, (limbs[10])*3+2], df1.iloc[:, (limbs[10])*3+3]],
    #                    [df1.iloc[:, (limbs[11])*3+1], df1.iloc[:, (limbs[11])*3+2], df1.iloc[:, (limbs[11])*3+3]]])
    # pts_3d = np.hstack(points)

    points = read_sensor(df1)
    pts_3d = np.hstack(
        np.array([[points[0][0], points[0][1], points[0][2]],
                  [points[1][0], points[1][1], points[1][2]],
                  [points[2][0], points[2][1], points[2][2]],
                  [points[3][0], points[3][1], points[3][2]],
                  [points[4][0], points[4][1], points[4][2]],
                  [points[5][0], points[5][1], points[5][2]],
                  [points[6][0], points[6][1], points[6][2]],
                  [points[7][0], points[7][1], points[7][2]],
                  [points[8][0], points[8][1], points[8][2]],
                  [points[9][0], points[9][1], points[9][2]],
                  [points[10][0], points[10][1], points[10][2]],
                  [points[11][0], points[11][1], points[11][2]]])).T

    points_cam = read_mp(df2, zscore=False)
    pts_2d = np.hstack(
        np.array([[points_cam[0][0], points_cam[0][1]],
                  [points_cam[1][0], points_cam[1][1]],
                  [points_cam[2][0], points_cam[2][1]],
                  [points_cam[3][0], points_cam[3][1]],
                  [points_cam[4][0], points_cam[4][1]],
                  [points_cam[5][0], points_cam[5][1]],
                  [points_cam[6][0], points_cam[6][1]],
                  [points_cam[7][0], points_cam[7][1]],
                  [points_cam[8][0], points_cam[8][1]],
                  [points_cam[9][0], points_cam[9][1]],
                  [points_cam[10][0], points_cam[10][1]],
                  [points_cam[11][0], points_cam[11][1]]])).T * size
    # # pixel to image
    # img_x = (pts_2d[:,0])/cam[0,0]
    # img_y = (pts_2d[:,1])/cam[1,1]
    # a = np.array([img_x,img_y])
    # b =pts_2d[:,0]
    pts_2d_resample = resample_poly(pts_2d, 60, 30)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pts_3d.T[0], pts_3d.T[1], pts_3d.T[2], marker='.')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # ax.set_xlim(-0.2, 0.8)
    # ax.set_zlim(0, 1.5)
    # ax.set_ylim(0, 1)
    # plt.savefig('3d_Sensor_all.jpg')
    plt.show()
    retval, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d_resample, cam, dist)
    P = rtvec_to_matrix(rvec, tvec)

    return pts_3d, P, rvec, tvec

def rtvec_to_matrix(rvec=(0, 0, 0), tvec=(0, 0, 0)):
    "Convert rotation vector and translation vector to 4x4 matrix"
    rvec = np.asarray(rvec)
    tvec = np.asarray(tvec)

    T = np.eye(4)
    (R, jac) = cv2.Rodrigues(rvec)
    T[:3, :3] = R
    T[:3, 3] = tvec.squeeze()
    return T
# 2d->3d projection
# 读取数据，并按身体部分分组

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

# dtw analysis

# frame to image

# Open the video file


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

def frame_diff(path_camera, path_sensor, axis, start, end, save=False):

    df = pd.read_csv(path_camera)
    video_name_list = df['filename'].unique()
    df1 = df.loc[df['filename'] == video_name_list[0]]
    df2 = df.loc[df['filename'] == video_name_list[1]]
    df3 = pd.read_csv(path_sensor)
    df3 = df3[start:end]

    # length = min(df1.shape[0], df2.shape[0],df3.shape[0])
    # df1 = df1.iloc[:length]
    # df2 = df2.iloc[:length]
    # df3 = df3.iloc[:length]

    # if axis == 'x':
    #     axis_mp= 'x'
    #     axis_sens= 'x'
    if axis == 'y':
        axis_mp = 'y'
        axis_sens = 'z'
    # elif axis == 'z':
    #     axis_mp = ''
    #     axis_sens = ''

    df1_diff = df1.iloc[:, df1.columns.str.endswith('.' + axis_mp)].diff(
        periods=1, axis=0).sum(axis=1)
    df2_diff = df2.iloc[:, df2.columns.str.endswith('.' + axis_mp)].diff(
        periods=1, axis=0).sum(axis=1)
    df3_diff = df3.iloc[:, df3.columns.str.endswith(' ' + axis_sens)].diff(
        periods=1, axis=0).sum(axis=1)

    x1_resampled = resample_poly(df1_diff, 60, 30)
    x2_resampled = resample_poly(df2_diff, 60, 30)
    x3_resampled = resample_poly(df3_diff, 60, 60)

    t1 = np.arange(x1_resampled.shape[0]) / 60.0  # Assuming 60 fps sample rate
    t2 = np.arange(x2_resampled.shape[0]) / 60.0
    t3 = np.arange(x3_resampled.shape[0]) / 60.0
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 9))
    # fig.suptitle('synchronisation by using {} '.format(NAME[i]))
    axs[0].plot(t1, x1_resampled, label='camera left')
    axs[0].set_ylabel('diff alone {} axis'.format(axis))
    axs[0].set_title('Before')
    axs[0].legend()
    axs[1].plot(t2, x2_resampled, label='camera right')
    axs[1].set_ylabel('diff alone {} axis'.format(axis))
    axs[1].legend()
    axs[2].plot(t3, x3_resampled, label='sensor')
    axs[2].set_ylabel('diff alone {} axis'.format(axis))
    axs[2].legend()
    # plt.savefig('diff_before.jpg')
    plt.show()

    corr12 = correlate(x2_resampled, x1_resampled, mode='full')
    shift12 = np.argmax(corr12) - (len(x1_resampled) - 1)
    corr13 = correlate(x3_resampled, x1_resampled, mode='full')
    shift13 = np.argmax(corr13) - (len(x1_resampled) - 1)
    corr23 = correlate(x3_resampled, x2_resampled, mode='full')
    shift23 = np.argmax(corr23) - (len(x2_resampled) - 1)

    # Apply time shifting to each signal to align them to the same starting point
    # if shift12 >= 0:
    #     x2_aligned = x2_resampled[shift12:]
    #     x1_aligned = x1_resampled[:len(x2_aligned)]
    #     df1 = df1[]
    # else:
    #     x1_aligned = x1_resampled[-shift12:]
    #     x2_aligned = x2_resampled[:len(x1_aligned)]

    # if shift13 >= 0:
    #     x3_aligned = x3_resampled[shift13:]
    #     # x1_aligned = x1_aligned[:len(x3_aligned)]
    #     x1_aligned = x1_resampled[:len(x3_aligned)]
    # else:
    #     # x1_aligned = x1_aligned[-shift13:]
    #     x1_aligned = x1_resampled[-shift13:]
    #     x3_aligned = x3_resampled[:len(x1_aligned)]

    # if shift23 >= 0:
    #     x3_aligned = x3_aligned[shift23:]
    #     x2_aligned = x2_aligned[:len(x3_aligned)]
    # else:
    #     x2_aligned = x2_aligned[-shift23:]
    #     x3_aligned = x3_aligned[:len(x2_aligned)]
    if shift13 >= 0:
        if shift23 >= 0:
            if shift13 >= shift23:
                x3_aligned = x3_resampled[shift13:]
                x2_aligned = x2_resampled[shift13 - shift23:]
                x1_aligned = x1_resampled[:min(len(x2_aligned), len(x3_aligned)
                                               )]
                df3_new = df3[shift13:]
            else:
                x3_aligned = x3_resampled[shift23:]
                x1_aligned = x1_resampled[shift23 - shift13:]
                x2_aligned = x2_resampled[:min(len(x1_aligned), len(x3_aligned)
                                               )]
                df3_new = df3[shift23:]
        elif shift23 < 0:
            x3_aligned = x3_resampled[shift13:]
            x2_aligned = x2_resampled[shift13 - shift23:]
            x1_aligned = x1_resampled[:min(len(x2_aligned), len(x3_aligned))]
            df3_new = df3[shift13:]
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

    # Trim the signals to the same length if necessary
    length = min(len(x1_aligned), len(x2_aligned), len(x3_aligned))
    x1_aligned = x1_aligned[:length]
    x2_aligned = x2_aligned[:length]
    x3_aligned = x3_aligned[:length]

    fig, axs = plt.subplots(nrows=3,
                            ncols=1,
                            figsize=(12, 9),
                            sharex=True,
                            sharey=True)
    # fig.suptitle('synchronisation by using {} '.format(NAME[i]))
    t4 = np.arange(x1_aligned.shape[0]) / 60.0
    axs[0].plot(t4, x1_aligned, label='camera left')
    axs[0].set_ylabel('diff alone {} axis'.format(axis))
    axs[0].set_title('After')
    axs[0].legend()
    axs[1].plot(t4, x2_aligned, label='camera right')
    axs[1].set_ylabel('diff alone {} axis'.format(axis))
    axs[1].legend()
    axs[2].plot(t4, x3_aligned, label='sensor')
    axs[2].set_ylabel('diff alone {} axis'.format(axis))
    axs[2].legend()
    # plt.savefig('diff_after1.jpg')
    plt.show()

    return shift13, shift23

def syn_mp(path):
    # path: csv file from mediapipe
    name1 = "pos_left.csv"
    name2 = "pos_right.csv"

    df = pd.read_csv(path)
    video_name_list = df['filename'].unique()

    df1 = df.loc[df['filename'] == video_name_list[0]]
    df2 = df.loc[df['filename'] == video_name_list[1]]

    NAME = [
        "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "RIGHT_SHOULDER",
        "RIGHT_ELBOW", "RIGHT_WRIST", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE",
        "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"
    ]
    # [11, 13, 15, 12, 14, 16, 23, 25, 27, 24, 26, 28]

    left_points = read_mp(df1, zscore=True)
    right_points = read_mp(df2, zscore=True)

    t_length = []
    new_x2 = []
    offset_all = []

    for i in range(0, 12):
        t_length.append(trajectory_length(left_points[i]))
    i = t_length.index(max(t_length))

    xyz_list = []

    length = min(len(left_points[i][0]), len(right_points[i][0]))
    x1 = left_points[i][0][:length]
    y1 = left_points[i][1][:length]
    z1 = left_points[i][2][:length]
    x2 = right_points[i][0][:length]
    y2 = right_points[i][1][:length]
    z2 = right_points[i][2][:length]

    corr_x = np.correlate(x1, x2, mode='full')
    corr_y = np.correlate(y1, y2, mode='full')
    corr_z = np.correlate(z1, z2, mode='full')

    dt = np.arange(1 - len(x1), len(x2))
    delay = dt[np.argmax(corr_y)]

    if delay >= 0:
        df2 = df2[delay:len(x2)]
        df1 = df1[0:len(x1) - delay]
    else:
        df2 = df2[0:len(x2) + delay]
        df1 = df1[-delay:len(x1)]
    # df1.to_csv(name1, index=False)
    # df2.to_csv(name2, index=False)
    # Find maximum correlation position
    lagx = (len(corr_x) - 1) // 2 - np.argmax(corr_x)
    x2_aligned = np.roll(x2, -lagx)
    lagy = (len(corr_y) - 1) // 2 - np.argmax(corr_y)
    y2_aligned = np.roll(y2, -lagy)
    lagz = (len(corr_z) - 1) // 2 - np.argmax(corr_z)
    z2_aligned = np.roll(z2, -lagz)
    # xyz_list.append(x2_aligned)
    # xyz_list.append(y2_aligned)
    # xyz_list.append(z2_aligned)
    # new_x2.append(xyz_list)

    # visualization
    t = np.arange(len(x1)) / 30.0  # Assuming 30 fps sample rate
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 9))
    # fig.suptitle('synchronisation by using {} '.format(NAME[i]))
    # axs[0, 0].plot(t, x1, label='x1')
    # axs[0, 0].plot(t, x2, label='x2')
    # axs[0, 0].set_ylabel('Position x ')
    # axs[0, 0].set_title('Original signals')
    # axs[0, 0].legend()

    # axs[0, 1].plot(t, x1, label='x1')
    # axs[0, 1].plot(t, x2_aligned, label='x2_aligned')
    # axs[0, 1].set_ylabel('X')
    # axs[0, 1].set_title('Aligned signals')
    # axs[0, 1].legend()

    # Second row of subplots: x3 vs x4
    axs[0].plot(t, y1, label='x1')
    axs[0].plot(t, y2, label='x2')
    axs[0].set_title('Original signals',fontsize = 18)
    axs[0].set_ylabel('Difference',fontsize = 18)
    axs[0].legend(fontsize = 18)
    axs[1].plot(t, y1, label='x1')
    axs[1].plot(t, y2_aligned, label='x2')
    axs[1].set_title('Aligned signals',fontsize = 18)
    axs[1].set_ylabel('Difference',fontsize = 18)
    axs[1].set_xlabel('Time (s)',fontsize = 18)
    axs[1].legend(fontsize = 18)

    # Third row of subplots: x5 vs x6
    # axs[2, 0].plot(t, z1, label='z1')
    # axs[2, 0].plot(t, z2, label='z2')
    # axs[2, 0].set_xlabel('Time (s)')
    # axs[2, 0].set_ylabel('Z')
    # axs[2, 0].legend()

    # axs[2, 1].plot(t, z1, label='z1')
    # axs[2, 1].plot(t, z2_aligned, label='z2_aligned')
    # axs[2, 1].legend()
    # axs[2, 1].set_xlabel('Time (s)')
    # axs[2, 1].set_ylabel('Z')
    # plt.savefig('synchronisation by using {} .jpg'.format(NAME[i]))
    for ax in axs:
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

    plt.show()

    return name1, name2

def syn2_mp(path):
    # path: csv file from mediapipe
    name1 = "pos_left.csv"
    name2 = "pos_right.csv"

    df = pd.read_csv(path)
    video_name_list = df['filename'].unique()

    df1 = df.loc[df['filename'] == video_name_list[0]]
    df2 = df.loc[df['filename'] == video_name_list[1]]

    NAME = [
        "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "RIGHT_SHOULDER",
        "RIGHT_ELBOW", "RIGHT_WRIST", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE",
        "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"
    ]
    # [11, 13, 15, 12, 14, 16, 23, 25, 27, 24, 26, 28]

    left_points = read_mp(df1)
    right_points = read_mp(df2)

    t_length = []
    new_x2 = []
    offset_all = []

    for i in range(0, 12):
        t_length.append(trajectory_length(left_points[i]))
        xyz_list = []

        length = min(len(left_points[i][0]), len(right_points[i][0]))
        x1 = left_points[i][0][:length]
        y1 = left_points[i][1][:length]
        z1 = left_points[i][2][:length]
        x2 = right_points[i][0][:length]
        y2 = right_points[i][1][:length]
        z2 = right_points[i][2][:length]

        corr_x = np.correlate(x1, x2, mode='full')
        corr_y = np.correlate(y1, y2, mode='full')
        corr_z = np.correlate(z1, z2, mode='full')

        dt = np.arange(1 - len(x1), len(x2))
        delay = dt[np.argmax(corr_y)]

        if delay >= 0:
            df2 = df2[delay:len(x2)]
            df1 = df1[0:len(x1) - delay]
        else:
            df2 = df2[0:len(x2) + delay]
            df1 = df1[-delay:len(x1)]
        # df1.to_csv(name1, index=False)
        # df2.to_csv(name2, index=False)
        # Find maximum correlation position
        lagx = (len(corr_x) - 1) // 2 - np.argmax(corr_x)
        x2_aligned = np.roll(x2, -lagx)
        lagy = (len(corr_y) - 1) // 2 - np.argmax(corr_y)
        y2_aligned = np.roll(y2, -lagy)
        lagz = (len(corr_z) - 1) // 2 - np.argmax(corr_z)
        z2_aligned = np.roll(z2, -lagz)
        # xyz_list.append(x2_aligned)
        # xyz_list.append(y2_aligned)
        # xyz_list.append(z2_aligned)
        # new_x2.append(xyz_list)
        offset_all.append(delay)

        # visualization
        t = np.arange(len(x1)) / 30.0  # Assuming 30 fps sample rate
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 9))
        fig.suptitle('synchronisation by using {} '.format(NAME[i]))
        axs[0, 0].plot(t, x1, label='x1')
        axs[0, 0].plot(t, x2, label='x2')
        axs[0, 0].set_ylabel('Position x ')
        axs[0, 0].set_title('Original signals')
        axs[0, 0].legend()

        axs[0, 1].plot(t, x1, label='x1')
        axs[0, 1].plot(t, x2_aligned, label='x2_aligned')
        axs[0, 1].set_ylabel('X')
        axs[0, 1].set_title('Aligned signals')
        axs[0, 1].legend()

        # Second row of subplots: x3 vs x4
        axs[1, 0].plot(t, y1, label='y1')
        axs[1, 0].plot(t, y2, label='y2')
        axs[1, 0].set_ylabel('Y')
        axs[1, 0].legend()

        axs[1, 1].plot(t, y1, label='y1')
        axs[1, 1].plot(t, y2_aligned, label='y2_aligned')
        axs[1, 1].set_ylabel('Y')
        axs[1, 1].legend()

        # Third row of subplots: x5 vs x6
        axs[2, 0].plot(t, z1, label='z1')
        axs[2, 0].plot(t, z2, label='z2')
        axs[2, 0].set_xlabel('Time (s)')
        axs[2, 0].set_ylabel('Z')
        axs[2, 0].legend()

        axs[2, 1].plot(t, z1, label='z1')
        axs[2, 1].plot(t, z2_aligned, label='z2_aligned')
        axs[2, 1].legend()
        axs[2, 1].set_xlabel('Time (s)')
        axs[2, 1].set_ylabel('Z')
        # plt.savefig('synchronisation by using {} .jpg'.format(NAME[i]))
        plt.show()
    i = t_length.index(max(t_length))

    return name1, name2, offset_all

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
# path_l = "F:\\studium\\Masterarbeit\\videos\\pos_left.mp4"
# path_r = "F:\\studium\\Masterarbeit\\videos\\pos_right.mp4"
# path = [path_l,path_r]
def calculate_sim(pts_sen,pts_cam):

    pts_sen = pts_sen.T
    _range_s_0= np.max(pts_sen[0])-np.min(pts_sen[0])
    _range_s_1= np.max(pts_sen[1])-np.min(pts_sen[1])
    _range_s_2= np.max(pts_sen[2])-np.min(pts_sen[2])
    pts_sen_x = (pts_sen[0]-np.min(pts_sen[0]))/ _range_s_0
    pts_sen_y = (pts_sen[1]-np.min(pts_sen[1]))/ _range_s_1
    pts_sen_z = (pts_sen[2]-np.min(pts_sen[2]))/ _range_s_2
    P_s = np.vstack([pts_sen_x,pts_sen_y,pts_sen_z])

    _range_c_0= np.max(pts_cam[0])-np.min(pts_cam[0])
    _range_c_1= np.max(pts_cam[1])-np.min(pts_cam[1])
    _range_c_2= np.max(pts_cam[2])-np.min(pts_cam[2])
    pts_cam_x = (pts_cam[0]-np.min(pts_cam[0]))/ _range_c_0
    pts_cam_y = (pts_cam[1]-np.min(pts_cam[1]))/ _range_c_1
    pts_cam_z = (pts_cam[2]-np.min(pts_cam[2]))/ _range_c_2
    P_c = np.vstack([pts_cam_x,pts_cam_y,pts_cam_z])


    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(P_c[0], P_c[1], P_c[2], marker='.')
    ax1.set_xlabel('X Label')
    ax1.set_ylabel('Y Label')
    ax1.set_zlabel('Z Label')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(P_s[0], P_s[1], P_s[2], marker='.')
    ax2.set_xlabel('X Label')
    ax2.set_ylabel('Y Label')
    ax2.set_zlabel('Z Label')
    # plt.savefig('compare_LMEDS')
    plt.show()


    return

mtx = np.array([[2.02354675e+03, 0.00000000e+00, 1.33796374e+03],
                [0.00000000e+00, 2.02505711e+03, 9.80799093e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array(
    [-0.30198934, -0.01203354, -0.00079829, -0.00049461, 0.07992867])
P1 = np.array([[-0.37144907,  0.9272702,  0.04685677, -0.31930919],
              [-0.23162339, -0.0436763, -0.97182456,  1.18659086],
              [-0.89909743, -0.37183645,  0.23100099,  0.39905917],
              [0.,  0.,  0.,  1.]])
P2 =np.array([[-0.12835448,  0.99144406,  0.02374452, -1.50651084],
       [ 0.02006785,  0.0265342 , -0.99944646,  0.61018696],
       [-0.99152529, -0.12780693, -0.02330194,  3.81251655],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
# mp_path = 'mediapipe_output.csv'
# leftpoints =read_mp('pos_left.csv')
# rightpoints =read_mp('pos_right.csv')
# print(leftpoints)
#

# sensor_data_path = 'Segment Position.csv'
# pts_3d, P, r, t = rot_trans_sen(sensor_data_path, "left.csv", mtx, dist, 20)

# print(mp_points[0][0],mp_points[0][1])
# print(keypoints_sens_3D[0][0],keypoints_sens_3D[0][1])
# 计算失真系数
# 去除畸变
# mediapipe得到2d points
# 同步，归一化
[a,b] = syn_mp("F:\\studium\\Masterarbeit\\videos\\pos_left_right_undist.csv")

# shift13,shift23 = frame_diff(
#     "F:\\studium\\Masterarbeit\\videos\\pos_left_right_undist.csv",
#     "F:\\studium\\Masterarbeit\\projectcode\\Segment Position.csv",
#     'y', 47200, 48720)

# print(offset)
# # 计算 E 得到Rt
#

# [a, b] = ["left.csv", "right.csv"]
# points_world,F, E, R, t, P1, P2, X = rot_trans_cam([a, b], mtx, dist,P2)

# calculate_sim(pts_3d,points_world)

#
# plot_sensor('Segment Position.csv', 47200, 48720)
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
# path = ["F:\\studium\\Masterarbeit\\videos\\reolink1-120056-121757-sebastian.mp4","F:\\studium\\Masterarbeit\\videos\\reolink2-120206-121909-sebastian.mp4"]
# undistortpath_l = undistort_video(path[0], mtx, dist, show=False, save=True, start_frame=1, end_frame=32000)
# # 报错，路径多加了一组引号
# undistortpath_r = undistort_video(path[1], mtx, dist, show=False, save=True, start_frame=1, end_frame=32000)
# undistortpath = [undistortpath_l,undistortpath_r]
# print(undistortpath)
# # #
# F= F_matrix(undistortpath)
# #
# R1,R2,t = rot_trans(mtx_l,mtx_r,F)
# Path to sensor data (.csv-data)
# Read the Excel file
# xlsx_file = pd.read_excel(sensor_data_path, sheet_name=None)
#
# # Loop through each sheet and save as csv
# for sheet_name, sheet_data in xlsx_file.items():
#     sheet_data.to_csv(f'{sheet_name}.csv', index=False)
# df = pd.read_excel(sensor_data_path,sheet_name=1)
# keypoints_sens_3D = fg.read_csv(sensor_data_path)
