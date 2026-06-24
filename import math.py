
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        # plt.show()
    i = t_length.index(max(t_length))

    return name1, name2, offset_all


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


[a,b,offset] = syn2_mp("F:\\studium\\Masterarbeit\\videos\\pos_left_right_undist.csv")