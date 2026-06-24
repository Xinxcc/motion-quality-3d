import pandas as pd
import numpy as np
from scipy.stats import zscore
from dtaidistance import dtw
from dtaidistance import dtw_ndim
from dtaidistance import dtw_visualisation as dtwvis
import matplotlib.pyplot as plt
from M import frame_diff, rot_trans_sen, rot_trans_cam, calculate_sim
# path and parameters
path_sensor = [r"F:\studium\Masterarbeit\X\Data\05302023_X_Biceps_sensor_Segment Position.csv",
               r"F:\studium\Masterarbeit\X\Data\05302023_X_Cor_Squat_sensor_Segment Position.csv",
               r"F:\studium\Masterarbeit\X\Data\05302023_X_Far_Squat_sensor_Segment Position.csv",
               r"F:\studium\Masterarbeit\X\Data\05302023_X_Rotation_sensor_Segment Position.csv",
               r"F:\studium\Masterarbeit\X\Data\05302023_X_Slight_Squat_sensor_Segment Position.csv"]
path_camera_landmarks = r"F:\studium\Masterarbeit\X\Data\X_data_undist_landmarks.csv"
path_camera_worldmarks = r"F:\studium\Masterarbeit\X\Data\X_data_undist_worldmarks.csv"

mtx = np.array([[2.02354675e+03, 0.00000000e+00, 1.33796374e+03],
                [0.00000000e+00, 2.02505711e+03, 9.80799093e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array(
    [-0.30198934, -0.01203354, -0.00079829, -0.00049461, 0.07992867])

# get the processed coordinates
def get_points(index_joint_left, path_camera_landmarks, path_sensor):
    """ use this function to get all 3d coordinates

    Args:
        index_joint_left (int): index of files
        path_camera_landmarks (str): path of landmarks 
        path_sensor (list): path of sensor data

    Returns:
        sen_: 3d points from sensor
        rec_: 3d points from 2d reconstruction
        left_: 3d points from single left camera
        right_: 3d points from single right camera
    """
    shift13, shift23, shift12 = frame_diff(index_joint_left,
                                           path_camera_landmarks, path_sensor[int(index_joint_left/2)], worldmarks=False)

    pts_3d, P_left, P_right, _, _, _, _ = rot_trans_sen(index_joint_left,
                                                        path_sensor[int(index_joint_left/2)], path_camera_landmarks, mtx, dist, shift13, shift23)

    points_world = rot_trans_cam(index_joint_left,
                                 path_camera_landmarks, path_sensor[int(index_joint_left/2)], mtx, dist, P_left, shift13, shift23)

    sen, rec, left, right = calculate_sim(index_joint_left,
                                          pts_3d, points_world, path_camera_worldmarks, path_sensor[int(index_joint_left/2)], shift13, shift23, P_left, P_right)

    len_sen = int(len(sen[0])/12)
    x1 = np.vstack([sen[0, i:i+len_sen]
                   for i in range(0, len(sen[0]), len_sen)])
    y1 = np.vstack([sen[1, i:i+len_sen]
                   for i in range(0, len(sen[0]), len_sen)])
    z1 = np.vstack([sen[2, i:i+len_sen]
                   for i in range(0, len(sen[0]), len_sen)])
    # Stack the divided arrays horizontally
    sen_ = np.vstack((x1, y1, z1)).T

    len_rec = int(len(rec[0])/12)
    x2 = np.vstack([rec[0, i:i+len_rec]
                   for i in range(0, len(rec[0]), len_rec)])
    y2 = np.vstack([rec[1, i:i+len_rec]
                   for i in range(0, len(rec[0]), len_rec)])
    z2 = np.vstack([rec[2, i:i+len_rec]
                   for i in range(0, len(rec[0]), len_rec)])
    # Stack the divided arrays horizontally
    rec_ = np.vstack((x2, y2, z2)).T

    len_left = int(len(left[0])/12)
    x3 = np.vstack([left[0, i:i+len_left]
                   for i in range(0, len(left[0]), len_left)])
    y3 = np.vstack([left[1, i:i+len_left]
                   for i in range(0, len(left[0]), len_left)])
    z3 = np.vstack([left[2, i:i+len_left]
                   for i in range(0, len(left[0]), len_left)])
    # Stack the divided arrays horizontally
    left_ = np.vstack((x3, y3, z3)).T

    len_right = int(len(right[0])/12)
    x4 = np.vstack([right[0, i:i+len_right]
                   for i in range(0, len(right[0]), len_right)])
    y4 = np.vstack([right[1, i:i+len_right]
                   for i in range(0, len(right[0]), len_right)])
    z4 = np.vstack([right[2, i:i+len_right]
                   for i in range(0, len(right[0]), len_right)])
    # Stack the divided arrays horizontally
    right_ = np.vstack((x4, y4, z4)).T

    return sen_, rec_, left_, right_


Cor_sen, Cor_rec, Cor_left, Cor_right = get_points(
    2, path_camera_landmarks, path_sensor)
Sli_sen, Sli_rec, Sli_left, Sli_right = get_points(
    8, path_camera_landmarks, path_sensor)
Far_sen, Far_rec, Far_left, Far_right = get_points(
    4, path_camera_landmarks, path_sensor)


joints_nummer = np.arange(0, 12)

torso = [0, 3, 6, 9]
left_arm = [0, 1, 2]
right_arm = [3, 4, 5]
left_leg = [6, 7, 8]
right_leg = [9, 10, 11]

limbs = [torso, left_arm, right_arm, left_leg, right_leg]
# limbs = [0,1,2,3,4,5,6,7,8,9,10,11]
######xin split frames
splits_cor = [96, 324, 566, 792, 1036, 1276, 1524, 1768, 2020, 2255, 2469]
splits_Far = [0, 234, 480, 708, 925, 1115, 1314, 1532, 1740, 1958, 2125]
splits_Sli = [43, 323, 568, 813, 1071, 1326, 1576, 1829, 2068, 2288, 2478]
######S split frames
# splits_cor = [29,276,512,717,933,1145,1353,1557,1781,1991,2219]
# splits_Far = [0,220,440,660,895,1126,1337,1553,1774,1998,2232]
# splits_Sli = [0,176,362,557,745,926,1110,1297,1466,1638,1840]
######P split frames
# splits_cor = [0,168,299,425,556,685,808,935,1059,1179,1303]
# splits_Far = [35,166,286,401,514,631,749,864,981,1100,1227]
# splits_Sli = [101,217,337,458,578,696,819,938,1056,1176,1301]


# calculate the standard mean value
# mean_for_correct_list = []
# for i in range(1,10):

#     m1 = Cor_rec[splits_cor[0]:splits_cor[1],:]
#     m2 = Cor_rec[splits_cor[i]:splits_cor[i+1],:]
#     md, mpaths = dtw_ndim.warping_paths(m1,m2)
#     best_mpath = dtw.best_path(mpaths)
#     idtw_list = []
#     for i in range(3):
#         for limb in limbs:
#             # 两个单个动作里每个器官的数据
#             si1 = []    #每个步长中动作1的x或者y坐标 size为步长*每个器官的维度（3，4，5）
#             si2 = []    #每个步长中动作1的x或者y坐标 size为步长*每个器官的维度（3，4，5）
#             # j最短路径的步长，0为s1的帧 1为s2的帧
#             for j in range(len(best_mpath)):
#                 si1.append(m1[best_mpath[j][0],12*i+np.array(limb)])
#                 si2.append(m2[best_mpath[j][1],12*i+np.array(limb)])
#             # si1 = s1[:,33*i+np.array(limb)]
#             # si2 = s2[:,33*i+np.array(limb)]
#             idtw_list.append(dtw_ndim.distance(si1,si2)) #对x和y的每一个器官得到一个距离，一共12个
#     mean_for_correct_list.append(idtw_list) # 一个动作存储15个距离，5个x，5个y, 5个z 对应5个器官 一共5行对应5个动作
# mean_for_correct = np.mean(np.array(mean_for_correct_list),axis=0)


#######Xin
# # mean value from cor_rec 
mean_for_correct = np.array([0.53514742, 0.6107502 , 0.49489429, 0.29720554, 0.47718837,
       0.44283578, 0.29408331, 0.48306522, 0.32143169, 0.39614678,
       0.18661852, 0.60480517, 0.65470566, 0.09689371, 0.09300049])
# # mean value from cor_sen
# mean_for_correct = np.array([0.77360284, 0.84963763, 0.9652781, 0.32297133, 0.29642906,
#                              0.63158738, 0.54862382, 0.66794645, 0.31584906, 0.30799892,
#                              0.18238723, 0.56401015, 0.5951923, 0.07134018, 0.06530222])

#######S
# # mean value from cor_rec 
# mean_for_correct = np.array([0.64678232, 0.67564279, 0.59173434, 0.53235723, 0.43617485,
#        0.71460985, 0.78763867, 0.83546622, 0.58213803, 0.38863078,
#        0.19853547, 0.62172588, 0.63664393, 0.09515054, 0.08841611])
# # mean value from cor_sen
# mean_for_correct =np.array([0.89912098, 1.0123624 , 0.92723582, 0.47604744, 0.34752254,
#        0.72405937, 0.89475215, 1.37232513, 0.5576163 , 0.40992576,
#        0.14924621, 0.69478082, 0.63047348, 0.0563916 , 0.04792294])


#######P
# # mean value from cor_rec 
# mean_for_correct = np.array([0.53763724, 2.82091968, 2.64522109, 0.41858569, 0.52627925,
#        0.81491772, 1.8493896 , 1.1474261 , 0.3260848 , 0.83465773,
#        0.38912948, 1.97594579, 2.11404664, 0.18110846, 0.28018645])
# # mean value from cor_sen
# mean_for_correct =np.array([0.70414065, 2.52922809, 2.57121035, 0.57048227, 0.68041729,
#        0.60818452, 1.05406213, 1.24875653, 0.33242463, 0.89512637,
#        0.17640236, 1.96926152, 2.0030037 , 0.0455896 , 0.05019605])

CSV_data = [1]*36

# calculate mDTW for normal squat
for t in range(1, 10):
    m1 = Cor_rec[splits_cor[0]:splits_cor[1], :]
    m2 = Cor_rec[splits_cor[t]:splits_cor[t+1], :]
    md, mpaths = dtw_ndim.warping_paths(m1, m2)
    best_mpath = dtw.best_path(mpaths)
    idtw_list = []
    for i in range(3):
        for limb in limbs:
            si1 = []
            si2 = []
            for j in range(len(best_mpath)):
                si1.append(m1[best_mpath[j][0], 12*i+np.array(limb)])
                si2.append(m2[best_mpath[j][1], 12*i+np.array(limb)])
            idtw_list.append(dtw_ndim.distance(si1, si2))
    normalized_idtw = np.around(idtw_list/mean_for_correct,3)
    
    # CSV_data = np.vstack((CSV_data,normalized_idtw.T))
    # f, ax = plt.subplots()
    # f.set_figheight(6)
    # f.set_figwidth(8)
    # f.tight_layout
    # width = 0.2
    # ax.bar(np.arange(0, 5), normalized_idtw[0:5], width=width)
    # ax.bar(np.arange(0, 5)+width, normalized_idtw[5:10], width=width)
    # ax.bar(np.arange(0, 5) + 2 * width, normalized_idtw[10:], width=width)
    # ax.set_xticks(np.arange(0, 5)+0.5*width,
    #               labels=['torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg'])
    # ax.set_ylim(0, 20)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # ax.legend(('x', 'y', 'z'), loc='upper left', fontsize=20)
    # # f.savefig(r"F:\studium\Masterarbeit\X" +
    # #           "\\Cor_camera" + "_sample" + str(t) + ".png")
    # plt.show()
    
    # # If you want to plot of each joint, choose limbs as imbs = [0,1,2,3,4,5,6,7,8,9,10,11]
    # f, ax = plt.subplots()
    # f.set_figheight(6)
    # f.set_figwidth(8)
    # f.tight_layout
    # width = 0.3

    # # Plotting the three sets of horizontal bar graphs
    # ax.barh(np.arange(0, 12), normalized_idtw[0:12], height=0.2)
    # ax.barh(np.arange(0, 12) + width, normalized_idtw[12:24], height=0.2)
    # ax.barh(np.arange(0, 12) + 2*width, normalized_idtw[24:], height=0.2)

    # # Setting custom tick labels on the y-axis
    # tick_positions = np.arange(0, 12) + 1.5 * width
    # tick_labels = ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "RIGHT_SHOULDER",
    #             "RIGHT_ELBOW", "RIGHT_WRIST", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE",
    #             "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"]
    # ax.set_yticks(tick_positions)
    # ax.set_yticklabels(tick_labels, fontsize=18)

    # # Setting the x-axis limits and tick font sizes
    # ax.set_xlim(0, 20)
    # ax.tick_params(axis='both', which='major', labelsize=18)

    # # Adding legend
    # ax.legend(('x', 'y', 'z'), loc='upper right', fontsize=18)

    # plt.show()

# calculate mDTW for shallow squat
# for t in range(0, 10):
#     m1 = Cor_rec[splits_cor[0]:splits_cor[1], :]
#     m2 = Sli_rec[splits_Sli[t]:splits_Sli[t+1], :]
#     md, mpaths = dtw_ndim.warping_paths(m1, m2)
#     best_mpath = dtw.best_path(mpaths)
#     idtw_list = []
#     for i in range(3):
#         for limb in limbs:
#             si1 = []
#             si2 = []
#             for j in range(len(best_mpath)):
#                 si1.append(m1[best_mpath[j][0], 12*i+np.array(limb)])
#                 si2.append(m2[best_mpath[j][1], 12*i+np.array(limb)])
#             idtw_list.append(dtw_ndim.distance(si1, si2))
#     normalized_idtw = np.around(idtw_list/mean_for_correct,3)
#     CSV_data = np.vstack((CSV_data,normalized_idtw.T))
#     f, ax = plt.subplots()
#     f.set_figheight(6)
#     f.set_figwidth(8)
#     f.tight_layout
#     width = 0.2
#     ax.bar(np.arange(0, 5), normalized_idtw[0:5], width=width)
#     ax.bar(np.arange(0, 5)+width, normalized_idtw[5:10], width=width)
#     ax.bar(np.arange(0, 5) + 2 * width, normalized_idtw[10:], width=width)
#     ax.set_xticks(np.arange(0, 5)+0.5*width,
#                   labels=['torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg'], fontsize=14)
#     ax.set_ylim(0, 20)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     ax.legend(('x', 'y', 'z'), loc='upper left', fontsize=18)
#     # f.savefig(r"F:\studium\Masterarbeit\X" +
#     #           "\\Sli_camera" + "_sample" + str(t) + ".png")
#     plt.show()

    # # If you want to plot of each joint, choose limbs as imbs = [0,1,2,3,4,5,6,7,8,9,10,11]
    # f, ax = plt.subplots()
    # f.set_figheight(6)
    # f.set_figwidth(8)
    # f.tight_layout
    # width = 0.3

    # Plotting the three sets of horizontal bar graphs
    # ax.barh(np.arange(0, 12), normalized_idtw[0:12], height=0.2)
    # ax.barh(np.arange(0, 12) + width, normalized_idtw[12:24], height=0.2)
    # ax.barh(np.arange(0, 12) + 2*width, normalized_idtw[24:], height=0.2)

    # # Setting custom tick labels on the y-axis
    # tick_positions = np.arange(0, 12) + 1.5 * width
    # tick_labels = ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "RIGHT_SHOULDER",
    #             "RIGHT_ELBOW", "RIGHT_WRIST", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE",
    #             "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"]
    # ax.set_yticks(tick_positions)
    # ax.set_yticklabels(tick_labels, fontsize=18)

    # # Setting the x-axis limits and tick font sizes
    # ax.set_xlim(0, 30)
    # ax.tick_params(axis='both', which='major', labelsize=18)
    # # Adding legend
    # ax.legend(('x', 'y', 'z'), loc='upper right', fontsize=18)

    # plt.show()

# # calculate mDTW for squat with feet too wide
# for t in range(0, 10):
#     m1 = Cor_rec[splits_cor[0]:splits_cor[1], :]
#     m2 = Far_rec[splits_Far[t]:splits_Far[t+1], :]
#     md, mpaths = dtw_ndim.warping_paths(m1, m2)
#     best_mpath = dtw.best_path(mpaths)
#     idtw_list = []
#     for i in range(3):
#         for limb in limbs:
#             si1 = []
#             si2 = []
#             for j in range(len(best_mpath)):
#                 si1.append(m1[best_mpath[j][0], 12*i+np.array(limb)])
#                 si2.append(m2[best_mpath[j][1], 12*i+np.array(limb)])
#             idtw_list.append(dtw_ndim.distance(si1, si2))
#     normalized_idtw = np.around(idtw_list/mean_for_correct,3)
#     CSV_data = np.vstack((CSV_data,normalized_idtw.T))
#     # f, ax = plt.subplots()
    # f.set_figheight(6)
    # f.set_figwidth(8)
    # f.tight_layout
    # width = 0.2
    # ax.bar(np.arange(0, 5), normalized_idtw[0:5], width=width)
    # ax.bar(np.arange(0, 5)+width, normalized_idtw[5:10], width=width)
    # ax.bar(np.arange(0, 5) + 2 * width, normalized_idtw[10:], width=width)
    # ax.set_xticks(np.arange(0, 5)+0.5*width,
    #               labels=['torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg'], fontsize=14)
    # ax.set_ylim(0, 20)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # ax.legend(('x', 'y', 'z'), loc='upper left', fontsize=18)
    # # f.savefig(r"F:\studium\Masterarbeit\X" +
    # #           "\\Far_camera" + "_sample" + str(t) + ".png")
    # plt.show()
    # # If you want to plot of each joint, choose limbs as imbs = [0,1,2,3,4,5,6,7,8,9,10,11]
    # f, ax = plt.subplots()
    # f.set_figheight(6)
    # f.set_figwidth(8)
    # f.tight_layout
    # width = 0.3

    # # Plotting the three sets of horizontal bar graphs
    # ax.barh(np.arange(0, 12), normalized_idtw[0:12], height=0.2)
    # ax.barh(np.arange(0, 12) + width, normalized_idtw[12:24], height=0.2)
    # ax.barh(np.arange(0, 12) + 2*width, normalized_idtw[24:], height=0.2)

    # # Setting custom tick labels on the y-axis
    # tick_positions = np.arange(0, 12) + 1.5 * width
    # tick_labels = ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "RIGHT_SHOULDER",
    #             "RIGHT_ELBOW", "RIGHT_WRIST", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE",
    #             "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"]
    # ax.set_yticks(tick_positions)
    # ax.set_yticklabels(tick_labels, fontsize=18)

    # # Setting the x-axis limits and tick font sizes
    # ax.set_xlim(0, 40)
    # ax.tick_params(axis='both', which='major', labelsize=18)

    # # Adding legend
    # ax.legend(('x', 'y', 'z'), loc='upper right', fontsize=18)

    # plt.show()

####save features
# Data = np.vstack((CSV_data.T,np.hstack(([0]*10,[1]*10,[2]*10)))).T
# df_tem = pd.DataFrame(data=Data)
# df = df_tem.drop([0])
# df.to_csv(r"F:\studium\Masterarbeit\X\Data\camera_features_3.csv")