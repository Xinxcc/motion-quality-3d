import numpy as np
from cmath import inf
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def camera_calibration(path, checkboard_size = (9, 6), square_size = 0.0262, show=False, save=False, frames=[]):
    # Define the number of inner corners of the checkerboard
    CHECKERBOARD_SIZE = checkboard_size

    # Define the size of each square in meters
    SQUARE_SIZE = square_size  # 26.2 mm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Load the video file
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count  = 0 
    # Arrays to store object points and image points from all the images
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane

    # Prepare the object points: (0,0,0), (1,0,0), ..., (8,5,0)
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

    # Create a new video writer to write the undistorted frames
    if save:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_name = path[:-4] + "_findboard.avi"        
        out = cv2.VideoWriter(output_name ,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width,height))
    while cap.isOpened():
        if len(frames)>0 and frame_count not in frames:
            if len(frames)>0 and frame_count > frames[-1]:
                print("End of selected segment reached!")
                break
            frame_count  = frame_count  + 1
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
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, flags = cv2.CALIB_CB_FAST_CHECK) # cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            # If the corners are found, add object points and image points
            if ret:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
                img_points.append(corners2)
            else:
                print("No checkerboard found!")
                pass
            if save:
                if ret:
                    cv2.drawChessboardCorners(frame, CHECKERBOARD_SIZE, corners2, ret)
                out.write(frame)
            if show:
                if ret:
                    cv2.drawChessboardCorners(frame, CHECKERBOARD_SIZE, corners2, ret)
                frame = cv2.resize(frame,(1280,960))
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Manual Interrupt!")
                    break
            frame_count  = frame_count  + 1

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
    return camera_matrix, distortion_coeffs

def undistort_video(path,camera_matrix, dist_coeffs, show=True, save=False, start_frame=0, end_frame=inf):
    # Load the camera matrix and distortion coefficients
    # camera_matrix = np.load('camera_matrix.npy')
    # dist_coeffs = np.load('distortion_coeffs.npy')

    # Load the video file
    cap = cv2.VideoCapture(path)
    frame_count  = 0 
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if save:
        output_name = path[:-4] + "_undistorted.avi"
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_name ,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (2394,1684))
    while cap.isOpened():
        if frame_count < start_frame:
            frame_count  = frame_count  + 1
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
            new_mtx,roi = cv2.getOptimalNewCameraMatrix(camera_matrix,dist_coeffs,(width,height),0.7,(width,height))
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs,None,new_mtx)
            undistorted = undistorted[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]
            if save:
                out.write(undistorted)
            if show:
                cv2.imshow('frame', undistorted)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Manual Interrupt!")
                    break
            frame_count  = frame_count  + 1

    # Release the video capture and writer objects
    cap.release()
    cv2.destroyAllWindows()
    return output_name


mtx = np.array([[2.02354675e+03, 0.00000000e+00, 1.33796374e+03],
       [0.00000000e+00, 2.02505711e+03, 9.80799093e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([-0.30198934, -0.01203354, -0.00079829, -0.00049461,  0.07992867])

cali_path_l = r"P:\SG4smartmedication\Physio_Aufnahme\Session_21_01\kamera_1\reolink1-144556-144650-calibration.mp4"
cali_path_r = r"P:\SG4smartmedication\Physio_Aufnahme\Session_21_01\kamera_2\reolink2-144704-144828-calibration.mp4"
cali_path = r"P:\SG4smartmedication\Physio_Aufnahme\calibration_reolink1-162316-162425.mp4"
# frame_list = list(range(0,150,8)) + list(range(210,720,8)) + list(range(810,1170,8)) + list(range(1230,1560,8)) + list(range(1620,1830,8)) + list(range(1860,1920,8)) + list(range(1950,2040,8))
# mtx, dist = camera_calibration(cali_path,frames=frame_list,show=True,save=True)
undistort_video(cali_path_l,mtx,dist,show=True,save=True,start_frame=1200,end_frame=1350)
undistort_video(cali_path_r,mtx,dist,show=True,save=True,start_frame=1230,end_frame=1380)