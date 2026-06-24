import os
from cmath import inf
import cv2
import mediapipe as mp
import numpy as np
import skimage.measure
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import pandas as pd
import matplotlib.pyplot as plt

def process_video(loc, sf = None, worldmarks = False, save = True, show = True, quality=720, lines=True, mediapipe_params = [False, 1, True, False, True, 0.5, 0.5], start_frame=0, end_frame=inf, entropy=False, threshold=1/3, wait_time=1/3):
    '''
    IN: 
        loc: Speicherort des Videos 
        sf: Ordner, in den das verarbeitete Video gespeichert werden soll
        worldmarks: Sollen 3D Weltkoordinaten statt 2D Bildkoordinaten ausgegeben werden?
        save: Soll Video gespeichert werden?
        show: Soll Video live gezeigt werden?
        quality: Höhe des ausgegebenen Videos in Pixel (720, 1080, 1440, ...)
        lines: Sollen die Verbindungslinien gezeichnet werden oder nur die Gelenke?
        mediapipe_params: Parameter, die an MediaPipe übergeben werden
        start_frame: Erstes Frame, das verarbeitet werden soll (geht bei 0 los)
        end_frame: Letztes Frame, das verarbeitet werden soll
        entropy: Soll auf die Entropie des Videos geachtet werden? (Frames werden nur bei genug Unterschied zu vorherigem Frame verarbeitet)
        threshold: Wie groß muss die Änderung der Entropie sein, um als "groß genug" zu gelten?
        wait_time: Nach welcher Zeit (in Bruchteil einer Sek.) wird die Entropie doch ignoriert?
    Return: 
        type: Art der zurückgegebenen Landmarks
        output_df: pandas.DataFrame mit Namen, Timestamps, Landmarktyp und Landmarks
    '''

    cap = cv2.VideoCapture(loc)

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_count  = 0 
    timestamps = []
    coordinates_list = []

    if entropy:
        en_seg = True
    else:
        en_seg = mediapipe_params[3]
    
    with mp_pose.Pose(static_image_mode=mediapipe_params[0],
                      model_complexity=mediapipe_params[1],
                      smooth_landmarks=mediapipe_params[2],
                      enable_segmentation=en_seg,
                      smooth_segmentation=mediapipe_params[4],
                      min_detection_confidence=mediapipe_params[5],
                      min_tracking_confidence=mediapipe_params[6],                   
                     ) as pose:
        
        # preparing the output
        width = int(cap.get(3))
        height = int(cap.get(4))
        # print(width,height)
        if save:
            if sf is not None:
                output_name = sf + loc[-6:-4] + "_output.avi"
            else:
                output_name = loc[:-4] + "_output.avi"
            while os.path.exists(output_name):
                output_name = output_name[:-4] + "_new.avi"
            out = cv2.VideoWriter(output_name ,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width,height))
            # out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc('X','V','I','D'), fps, (width, height))
        scale = False
        if height < quality:
            scale = True
            width = int(quality/height * width)
            height = quality

        previous_masked_frame = np.empty((height,width,3))
        freeze_counter = 0

        while cap.isOpened():
            if frame_count < start_frame:
                frame_count  = frame_count  + 1
                cap.grab()
                pass
            elif frame_count > end_frame:
                break
            else:
                ret,frame = cap.read()
                # # 调整窗口大小
                # cv2.namedWindow('Mediapipe',0)
                # cv2.resizeWindow('Mediapipe', 540, 960)
                if ret == True:
                    if scale:
                        frame = cv2.resize(frame, (width,height), interpolation=cv2.INTER_CUBIC)  # if computational power available: interpolation=cv2.INTER_CUBIC
                    # Konvertieren von RGB zu BGR
                    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    frame.flags.writeable = False
                    # Pose Estimation
                    results = pose.process(frame)
                    # Recolor back to BGR
                    frame.flags.writeable = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Get segmentation mask if entropy is calculated
                    if entropy:
                        seg_mask = results.segmentation_mask
                        if seg_mask.any() != None:
                            mask = np.stack((seg_mask,) * 3, axis=-1) > 0.1
                            masked_frame = np.multiply(frame, mask)
                            difference = masked_frame-previous_masked_frame
                            shannon_entropy = skimage.measure.shannon_entropy(difference)
                            previous_masked_frame = masked_frame
                            # plt.imshow(np.sum(difference[:,:,:],2), interpolation='nearest')


                    # Extract landmarks
                    t = round(frame_count/fps*10**3,2)
                    timestamps.append(t)
                    
                    if not entropy or (entropy and (shannon_entropy > threshold or freeze_counter >= fps*wait_time)):
                        freeze_counter = 0
                        xyz_list = []
                        # Xw
                        if (results.pose_world_landmarks != None and worldmarks):
                            for idx, landmark in enumerate(results.pose_world_landmarks.landmark):
                                xyz_list.append(landmark.x)
                                xyz_list.append(landmark.y)
                                xyz_list.append(landmark.z)
                                xyz_list.append(landmark.visibility)

                        elif (results.pose_landmarks != None and not worldmarks):
                            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                                xyz_list.append(landmark.x)
                                xyz_list.append(landmark.y)
                                xyz_list.append(landmark.z)
                                xyz_list.append(landmark.visibility)
                        else:
                            print("No landmark available at " + str(t))
                            pass
                        coordinates_list.append(xyz_list)
                    else:
                        freeze_counter += 1
                        coordinates_list.append(coordinates_list[-1])
                    # Render 2D pose landmarks
                    connection_lines = None
                    if lines:
                        connection_lines = mp_pose.POSE_CONNECTIONS
                    r = int(max(min(width, height)/300,1))
                    thick = int(max(min(width, height)/250,r+1))
                    mp_drawing.draw_landmarks(frame,results.pose_landmarks,connection_lines,
                                              mp_drawing.DrawingSpec(color=(0, 0, 255),thickness=thick,circle_radius=r)
                                            )        
                    if save:
                        out.write(frame)
                    if show:
                      #  cv2.resizeWindow('Mediapipe',540,960)
                        cv2.imshow('Mediapipe',frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break           
                    frame_count  = frame_count  + 1
                else:
                    break
        cap.release()
        cv2.destroyAllWindows()

    output_df = pd.DataFrame(coordinates_list)
    body_parts = ["NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", 
    "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"]
    column_names = []
    for i in body_parts:
        column_names.append(i+".x")
        column_names.append(i+".y")
        column_names.append(i+".z")
        column_names.append(i+".visibility")
    output_df = output_df.set_axis(column_names,axis=1)

    output_df.insert(0,"timestamp [ms]", timestamps)
    output_df.insert(0,"filename",loc) 
    if worldmarks:
        type = "world"
    else:
        type = "camera"    
    
    return type, output_df