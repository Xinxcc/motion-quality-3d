import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from estimation_mediapipe_pose import process_video
#from estimation_mediapipe_holistic import process_video_holistically
from cmath import inf
# from mediapipe_plot_function import plot_from_csv

# Enter path to videos here
# video_folder = r"P:\SG4smartmedication\Physio_Aufnahme\Session_21_01\animations"# "R:\\InternalDatabases\\Locomotion Lab\\angle_estimation\\" # "R:\InternalDatabases\Locomotion Lab\\test\\" "R:\\InternalDatabases\\Locomotion Lab\\rgb_videos\\"
video_folder = r"F:\\studium\\Masterarbeit\\P\\Videos\\"
save_folder = video_folder
all_videos = True
video_type = ".avi"
holistic = False

if all_videos:
    video_names = os.listdir(video_folder)
    list_of_names = [os.path.join(video_folder, f)
                     for f in video_names if f.endswith(video_type)]
else:
    # specify video details
    video_names = ["F:\\studium\\Masterarbeit\\S\\Videos\\05302023_S_Biceps_lef_undistorted.avi",
                   "F:\\studium\\Masterarbeit\\S\\Videos\\05302023_S_Biceps_righ_undistorted.avi",
                   "F:\\studium\\Masterarbeit\\S\\Videos\\05302023_S_Cor_Squat_lef_undistorted.avi",
                   "F:\\studium\\Masterarbeit\\S\\Videos\\05302023_S_Cor_Squat_righ_undistorted.avi",
                   "F:\\studium\\Masterarbeit\\S\\Videos\\05302023_S_Far_Squat_lef_undistorted.avi",
                   "F:\\studium\\Masterarbeit\\S\\Videos\\05302023_S_Far_Squat_righ_undistorted.avi",
                   "F:\\studium\\Masterarbeit\\S\\Videos\\05302023_S_Rotation_lef_undistorted.avi",
                   "F:\\studium\\Masterarbeit\\S\\Videos\\05302023_S_Rotation_righ_undistorted.avi",
                   "F:\\studium\\Masterarbeit\\S\\Videos\\05302023_S_Slight_Squat_lef_undistorted.avi",
                   "F:\\studium\\Masterarbeit\\S\\Videos\\05302023_S_Slight_Squat_righ_undistorted.avi"]  # ["aufnahme1_fps","aufnahme2_fps","aufnahme3_fps","aufnahme4_fps"]# ["aufnahme4_fps"]
    list_of_names = []
    for i in video_names:
        file = video_folder + "\\" + str(i) + video_type
        list_of_names.append(file)

relevant_frame_list = [[0, inf] for i in range(len(list_of_names))]
# relevant_frame_list[0] = [9000, 24000] # for aufnahme 1
# relevant_frame_list[0] = [3700, 18400] # for aufnahme 1
# relevant_frame_list[1] = [3600, 18200] # for aufnahme 3
# relevant_frame_list[3] = [3700, 18300] # for aufnahme 4

# setting mediapipe parameters
med_par = []
# set static_image_mode (default: False); set to True if person detector is supposed to be done on every image (e.g. unrelated images instead of video); setting this to True leads to ignoring smooth_landmarks, smooth_segmentation and min_tracking_confidence
med_par.append(False)
# set model_complexity (default: 1, possible 0-2)
med_par.append(2)
# set smooth_landmarks (default: True); filters landmark positions; overruled by static image mode
med_par.append(True)
# set enable_segmentation (default: False); would also return segmentation mask additional to landmarks; Overruled, when entropy is given to process_video
med_par.append(False)
# set smooth_segmentation (default: True); filters segmentation mask; ignored when segmentation not enabled or static_image_mode=True
med_par.append(True)
# ONLY IF HOLISTIC: set refine_face_landmarks (default: False); True means that landmarks in the face are more refined, with additional landmarks for iris, around eyes and lips
if holistic:
    med_par.append(False)
# set min_detection_confidence (default: 0.5); Minimum confidence value from the person-detection model for the detection to be considered successful
med_par.append(0.5)
# set min_tracking_confidence (default: 0.5); Minimum confidence value from the landmark-tracking model for the pose landmarks to be considered tracked successfully
# otherwise: person detection will be invoked on the next input image. Setting it to a higher value can increase robustness of the solution, at the expense of a higher latency
med_par.append(0.4)

print("-----------Starting import---------")

df = pd.DataFrame()
# Importieren und Anwendung von MediaPipe
for counter, name in enumerate(list_of_names):
    # if holistic:
    # type, coor_df = process_video_holistically(
    #     name, sf = save_folder, worldmarks = True, save = False, show = True,
    #     quality=720, lines=True, mediapipe_params=med_par,
    #     start_frame=relevant_frame_list[counter][0], end_frame=relevant_frame_list[counter][1],
    #     entropy=False, threshold=1/3, wait_time=1/3)

    type, coor_df = process_video(
        name, sf=save_folder, worldmarks=False, save=False, show=False,
        quality=720, lines=True, mediapipe_params=med_par,
        start_frame=relevant_frame_list[counter][0], end_frame=relevant_frame_list[counter][1],
        entropy=False, threshold=1/3, wait_time=1/3)
    df = df.append(coor_df)
    print(name + str("..... Done"))

print("------------FINISHED---------------")

save_csv = True
if save_csv:
    name_to_save = os.path.join(save_folder, "P_data_undist_landmarks.csv")
    while os.path.exists(name_to_save):
        name_to_save = name_to_save[:-4] + "_new.csv"
    df.to_csv(name_to_save, index=False)
