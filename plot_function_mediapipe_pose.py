from cmath import inf
import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def plot_from_csv(path_csv, sf = None, with_video=True, quality = 720, save=False, show=True, lines=True, output_suffix="_drawing_output.avi", color_coding='depth'):
    '''
    IN: 
        path_csv: Speicherort der csv (nur eine, kann Infos zu mehreren Videos enthalten)
        sf: Ordner, in den das verarbeitete Video gespeichert werden soll
        with_video: soll Plot als Maske über das Video gelegt werden?
        loc: Liste der Speicherorte der Videos
        quality: Höhe des ausgegebenen Videos in Pixel (480, 720, 1080, 1440)
        save: Soll Video gespeichert werden?
        show: Soll Video live gezeigt werden?
        lines: Sollen Verbindungslinien gezeichnet werden?
        output_suffix: Suffix des Namens, mit dem das Video gespeichert werden soll. Muss Dateiendung enthalten.
        color_coding: welche Größe soll durch die Farbe kodiert werden? Mögliche Angaben: 'visibility' und 'depth'
    '''
    df = pd.read_csv(path_csv)
    video_name_list = df['filename'].unique()
    for idx, video_name in enumerate(video_name_list):
        current_df = df.loc[df['filename'] == video_name]
        current_df = current_df.drop('filename',axis=1)
        delta_t = current_df['timestamp [ms]'].iloc[1]-current_df['timestamp [ms]'].iloc[0]
        current_df['frame_number'] = (current_df['timestamp [ms]']/delta_t).astype('int')
        start_frame = current_df['frame_number'].iloc[0]
        end_frame = current_df['frame_number'].iloc[-1]
        cap = cv2.VideoCapture(video_name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(3))
        height = int(cap.get(4))
        scale = False
        if height < quality:
            scale = True
            width = int(quality/height * width)
            height = quality
        if not with_video:
            width = int(16/9*quality)
            height = quality

        if save:
            if sf is not None:
                output_name = sf + video_name[-6:-4] + output_suffix
            else:
                output_name = video_name[:-4] + output_suffix
            while os.path.exists(output_name):
                output_name = output_name[:-4] + "_new.avi"
            out = cv2.VideoWriter(output_name ,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width,height))
        frame_count  = 0
        csv_count = 0

        color_coding_variable = 4
        if color_coding == 'depth':
            # normalize z coordinates to the range of [0 1] to color code it
            color_coding_variable = 3
            min_depth = inf
            max_depth = 0
            for c in current_df.columns:
                if c[-2:] == '.z':
                    current_min = np.min(current_df[c])
                    current_max = np.max(current_df[c])
                    if current_min < min_depth:
                        min_depth = current_min
                    if current_max > max_depth:
                        max_depth = current_max
            for c in current_df.columns:
                if c[-2:] == '.z':
                    current_df[c] = (np.array(current_df[c])-min_depth)/(max_depth-min_depth)


        # TIEFPASS!
        # sig = current_df.loc[:,'RIGHT_WRIST.x']
        # sig_f = np.fft.fft(np.array(sig))
        # tp_filter = signal.butter(2, 10, 'lp', fs=fps, output='sos')
        # filtered = signal.sosfilt(tp_filter, current_df.loc[:,'RIGHT_WRIST.x'])
        # filtered_f = fft(filtered)
        # plt.plot(sig_f)
        # plt.show()
        # plt.plot(filtered_f)

        if with_video:
            while cap.isOpened():
                if frame_count < start_frame:
                    frame_count  += 1
                    cap.grab()
                    pass
                elif frame_count > end_frame:
                    break
                else:
                    ret,frame = cap.read()  
                    if ret == True:
                        if scale:
                            frame = cv2.resize(frame, (width,height), interpolation=cv2.INTER_CUBIC)  # if computational power available: interpolation=cv2.INTER_CUBIC
                        if not current_df.iloc[csv_count].isnull().values.any():
                            if lines:
                                for i in mp_pose.POSE_CONNECTIONS:
                                    start_point = tuple(np.multiply(np.array([current_df.iloc[csv_count].iloc[i[0]*4+1],current_df.iloc[csv_count].iloc[i[0]*4+2]]), [width, height]).astype(int))
                                    end_point = tuple(np.multiply(np.array([current_df.iloc[csv_count].iloc[i[1]*4+1],current_df.iloc[csv_count].iloc[i[1]*4+2]]), [width, height]).astype(int))
                                    thickness = int(max(min(width, height)/250,1))
                                    cv2.line(frame, start_point, end_point, (200, 200, 200), thickness)
                            for j in range(33):
                                drawing_coordinates = tuple(np.multiply(np.array([current_df.iloc[csv_count].iloc[j*4+1],current_df.iloc[csv_count].iloc[j*4+2]]), [width, height]).astype(int))
                                radius = int(max(min(width, height)/100,1))
                                blue_part = int(255*current_df.iloc[csv_count].iloc[j*4+color_coding_variable])
                                green_part = int(255*(1-current_df.iloc[csv_count].iloc[j*4+color_coding_variable]))
                                red_part = 0
                                cv2.circle(frame, drawing_coordinates, radius, (blue_part, green_part, red_part), -1) # green = (0, 100, 0)
                        if save:
                            out.write(frame)
                        if show:
                            cv2.imshow('Mediapipe',frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break     
                        frame_count  += 1
                        csv_count += 1
                    else:
                        break


        else:
            for csv_count in range(end_frame+1-start_frame):
                frame = np.ones((height,width,3),dtype=np.uint8)
                if not current_df.iloc[csv_count].isnull().values.any():
                    if lines:
                        for i in mp_pose.POSE_CONNECTIONS:
                            start_point = tuple(np.multiply(np.array([current_df.iloc[csv_count].iloc[i[0]*4+1],current_df.iloc[csv_count].iloc[i[0]*4+2]]), [width, height]).astype(int))
                            end_point = tuple(np.multiply(np.array([current_df.iloc[csv_count].iloc[i[1]*4+1],current_df.iloc[csv_count].iloc[i[1]*4+2]]), [width, height]).astype(int))
                            thickness = int(max(min(width, height)/250,1))
                            cv2.line(frame, start_point, end_point, (200, 200, 200), thickness)
                    for j in range(33):
                        drawing_coordinates = tuple(np.multiply(np.array([current_df.iloc[csv_count].iloc[j*4+1],current_df.iloc[csv_count].iloc[j*4+2]]), [width, height]).astype(int))
                        radius = int(max(min(width, height)/100,1))
                        blue_part = int(255*current_df.iloc[csv_count].iloc[j*4+color_coding_variable])
                        green_part = int(255*(1-current_df.iloc[csv_count].iloc[j*4+color_coding_variable]))
                        red_part = 0
                        cv2.circle(frame, drawing_coordinates, radius, (blue_part, green_part, red_part), -1) # green = (0, 100, 0)
                if save:
                    out.write(frame)
                if show:
                    cv2.imshow('Mediapipe',frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break     

csv_path = "E:\\anonymisation\mediapipe_holistic_output_GTKA.csv" # "R:\InternalDatabases\Locomotion Lab\\test\mediapipe_output.csv"
#csv_path = "F:\studium\Masterarbeit\projectcode\mediapipe_output.csv"
save_folder = "E:\\anonymisation\\"
#save_folder = "F:\\studium\Masterarbeit\projectcode\\"
plot_from_csv(csv_path, sf=save_folder,with_video=False, quality=480, save=True, lines=True, color_coding='visibility')