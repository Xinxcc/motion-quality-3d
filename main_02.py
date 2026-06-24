import cv2
import os
import numpy as np
from Class_MP import Detector_MP
from Class_OP import Detector_OP
import functions_general as fg
import transformation as tf

process_img = False                                                     # Want to process a video (False) or images (True)?
init_point_by_mouse_click = False                                       # Initial points get by mouseclick (True) or by OpenPose (False)

if process_img:
    path = "Data/ME/"                                                   # Path where images are saved
    image_names = os.listdir(path)                                      # List of all images in given path
else:
    path = "Data/"                                                      # Path where video is saved
    video_name = "Kwin-001.mp4"                                    # Video name
    sensor_data_path = 'Data/Sensor_Data/Kwin-001.csv'             # Path to sensor data (.csv-data)
    keypoints_sens_3D = fg.read_csv(sensor_data_path)                   # Reading keypoints of sensors (1713 frames)
    # print(keypoints_sens_3D)

detector_MP = Detector_MP()                                             # Create instance of Mediapipe pose Detector
# detector_OP = Detector_OP(path=path, process_img=process_img)           # Create instance of OpenPose pose Detector

# Process and display images
if process_img:
    for image_name in image_names:
        imagePath = os.path.join(path, image_name)                      # Append current image to path
        imageToProcess = cv2.imread(imagePath)
        img = fg.scale_img(imageToProcess, 0.5, 0, 0)                   # Scale image if needed (img, scalefactor, width, height)

        # keypoints_op = detector_OP.findKeypoints(img)                   # Keypoints of OpenPose
        keypoints_mp = detector_MP.findKeypoints(img, draw=False)       # Keypoints of Mediapipe, if second input True --> Draw skeleton of MP (not working)

        fg.draw_keypoints(img, keypoints_mp, [])              # Visual comparison between both models

        # print("Body keypoints Mediapipe: \n" + str(keypoints_mp))
        # print("Body keypoints OpenPose: \n" + str(keypoints_op))

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

# Process and display videos
else:
    cap = cv2.VideoCapture(path+video_name)
    ret, frame = cap.read()
    count = 0  # No synchronisation just start at given sensor datapoint

    if init_point_by_mouse_click:                           # Setting init markers for Transformation manual
        init_points, count = fg.mousePoints(cap, count)
    else:                                                   # Setting init markers for Transformation via OP
        while ret:  # Find suitable initial image in video
            cv2.imshow('Find Points', frame)
            if cv2.waitKey(0) & 0xFF == ord('n'):
                ret, frame = cap.read()
                count += 1
            else:
                cv2.destroyWindow('Find Points')
                break
        # keypoints_op = detector_OP.findKeypoints(frame)
        # kp_op = fg.important_kp_op(keypoints_op)
        keypoints_mp = detector_MP.findKeypoints(frame)
        kp_mp = fg.important_kp_mp(keypoints_mp)
        init_points = fg.get_init_points(kp_mp)

        fg.draw_keypoints_np_array(frame, init_points, init_points, init_points, True)
        cv2.waitKey(0)


    # Computing matrices (rotation, translation) for perspective projection
    kp_sens_3D = fg.important_kp_sens(keypoints_sens_3D, count)
    rot, trans, cam, dist = tf.get_rot_trans(kp_sens_3D, init_points, frame)


    while cap.isOpened():
        count += 1

        ret, frame = cap.read()
        if ret == True:

            # Undistort current frame
            # h, w = frame.shape[:2]
            # newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cam, dist, (w, h), 0, (w, h))
            # frame = cv2.undistort(frame, cam, dist, None, newCameraMatrix)

            # keypoints_op = detector_OP.findKeypoints(frame)
            keypoints_mp = detector_MP.findKeypoints(frame, draw=False)

            # kp_op = fg.important_kp_op(keypoints_op)
            kp_mp = fg.important_kp_mp(keypoints_mp)

            kp_sens_3D = fg.important_kp_sens(keypoints_sens_3D, count)
            kp_sens_2D = tf.space_to_plane(kp_sens_3D, rot, trans, cam, dist)   # Transforamtion


            fg.draw_keypoints_np_array(frame, kp_mp, np.array([]), kp_sens_2D)  # Visual comparison between both models and sensors

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else: break

print("Successfully finished.")