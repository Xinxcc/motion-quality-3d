import cv2
import csv
import numpy as np

counter = 0  # Counter for function mouse clicks

'''
Scaling the shape of an image. First option via scaling factor. If the scaling factor is 0 via absolute values
Inputs:
    img:            Image to be resized
    scale_factor:   Resizing factor
    width:          Absolute width
    height:         Absolute Height
Output:
    img_resized:    Resized image
'''


def scale_img(img, scale_factor=0.5, width=0, height=0):
    if scale_factor != 0:
        scaled_width = int(img.shape[1] * scale_factor)
        scaled_height = int(img.shape[0] * scale_factor)

        img_resized = cv2.resize(img, (scaled_width, scaled_height))

    else:
        img_resized = cv2.resize(img, (int(width), int(height)))

    return img_resized


'''
Drawing keypoints (list) on given image
Inputs:
    img:        Image to draw on
    list_mp:    Keypoints of mediapipe model
    list_op:    Keypoints of openpose model
'''
def draw_keypoints_list(img, list_mp, list_op):
    for i in range(len(list_mp)):
        cv2.circle(img, (list_mp[i][1], list_mp[i][2]), 4, (0, 0, 255), cv2.FILLED)
        cv2.putText(img, 'mediapipe', (20, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    for i in range(len(list_op)):
        cv2.circle(img, (list_op[i][1], list_op[i][2]), 4, (255, 0, 0), cv2.FILLED)
        cv2.putText(img, 'openpose', (20, 90), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('Comparison MP and OP', img)


def draw_keypoints_np_array(img, array_mp, array_op, array_sens, show_init_pic = False):
    '''
    Drawing keypoints (np-array) on given image
    :param img: actual frame
    :param array_mp: Important keypoints of mediapipe model
    :param array_op: Important keypoints of openpose model
    '''
    for i in range(array_mp.shape[0]):
        cv2.circle(img, (int(array_mp[i][0]), int(array_mp[i][1])), 4, (0, 0, 255), cv2.FILLED)
        cv2.putText(img, 'mediapipe', (20, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    if len(array_op) != 0:
        for i in range(array_op.shape[0]):
            cv2.circle(img, (int(array_op[i][0]), int(array_op[i][1])), 4, (255, 0, 0), cv2.FILLED)
            cv2.putText(img, 'openpose', (20, 90), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    if not show_init_pic:
        for i in range(array_sens.shape[0]):
            cv2.circle(img, (int(array_sens[i][0][0]), int(array_sens[i][0][1])), 4, (0, 255, 0), cv2.FILLED)
            cv2.putText(img, 'sensors', (20, 140), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv2.imshow('Comparison MP and OP and sensors', img)


def mouseClick(event, x, y, flags, params):
    global counter

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_xy[counter] = x, y
        counter += 1
        # print(mouse_xy)


'''
Getting Coordinates in a frame over mouseclick
Input:
    video:      Video which will be processed
Outputs:
    mouse_xy:   Coordinates marked with mouseclick
'''
def mousePoints(video, count):
    global mouse_xy
    mouse_xy = np.zeros((12, 2), np.int)

    ret, frame = video.read()

    print('Select suitable initial frame. Press "n" for next frame and "q" if optimal frame was found')
    while ret:  # Find suitable initial image in video
        cv2.imshow('Find Points', frame)
        if cv2.waitKey(0) & 0xFF == ord('n'):
            ret, frame = video.read()
            count += 1
            print(count)
        else:
            cv2.destroyWindow('Find Points')
            break

    print('Now select important points')
    print('1. Click on left shoulder joint')
    print('2. Click on right shoulder joint')
    print('3. Click on left hip joint')
    print('4. Click on right hip joint')
    print('5. Click on left elbow joint')
    print('6. Click on right elbow joint')
    print('7. Click on left knee joint')
    print('8. Click on right knee joint')
    print('9. Click on left wrist joint')
    print('10. Click on right wrist joint')
    print('11. Click on left ankle joint')
    print('12. Click on right ankle joint')

    while ret:  # Select important points
        for x in range(0, 12):
            cv2.circle(frame, (mouse_xy[x][0], mouse_xy[x][1]), 3, (0, 255, 0), cv2.FILLED)

        cv2.imshow('Select Points', frame)
        cv2.setMouseCallback('Select Points', mouseClick)
        cv2.waitKey(1)

        if counter == 12:
            break

    cv2.destroyWindow('Select Points')

    return mouse_xy, count


'''
Get the sensor data and return it
'''
def read_csv(path):
    keypoints = np.empty((0, 69))
    count_i = 0
    count_j = 0
    with open(path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')

        next(csv_reader)  # Skip first line (Header)

        for line_string in csv_reader:
            if (count_j % 2) == 0:  # Für 60 fps
                line_float = []
                for i in range(len(line_string)):  # Changing the list elements from string to float
                    line_float.append(float(line_string[i].replace(',', '.')))

                del line_float[0]  # delete frame number
                keypoints = np.vstack((keypoints, line_float))
                count_i += 1

            count_j += 1

    # print(keypoints[0,:])
    return keypoints


def important_kp_op(kp_op):
    '''
    Sort keypoints of open pose in given order (left arm, right arm, left leg, right leg) and safes into numpay array
    :param kp_op: keypoints of open pose
    :return: kp: sorted list
    '''
    kp = np.zeros((14, 2))
    if kp_op:
        kp[0][0] = kp_op[5][1]
        kp[0][1] = kp_op[5][2]
        kp[1][0] = kp_op[6][1]
        kp[1][1] = kp_op[6][2]
        kp[2][0] = kp_op[7][1]
        kp[2][1] = kp_op[7][2]
        kp[3][0] = kp_op[2][1]
        kp[3][1] = kp_op[2][2]
        kp[4][0] = kp_op[3][1]
        kp[4][1] = kp_op[3][2]
        kp[5][0] = kp_op[4][1]
        kp[5][1] = kp_op[4][2]
        kp[6][0] = kp_op[12][1]
        kp[6][1] = kp_op[12][2]
        kp[7][0] = kp_op[13][1]
        kp[7][1] = kp_op[13][2]
        kp[8][0] = kp_op[14][1]
        kp[8][1] = kp_op[14][2]
        kp[9][0] = kp_op[19][1]
        kp[9][1] = kp_op[19][2]
        kp[10][0] = kp_op[9][1]
        kp[10][1] = kp_op[9][2]
        kp[11][0] = kp_op[10][1]
        kp[11][1] = kp_op[10][2]
        kp[12][0] = kp_op[11][1]
        kp[12][1] = kp_op[11][2]
        kp[13][0] = kp_op[22][1]
        kp[13][1] = kp_op[22][2]
    return kp


def important_kp_mp(kp_mp):
    '''
    Sort keypoints of mediapipe in given order (left arm, right arm, left leg, right leg) and safes into numpay array
    :param kp_mp: keypoints of open pose
    :return: kp: sorted list
    '''
    kp = np.zeros((14, 2))
    if kp_mp:
        kp[0][0] = kp_mp[11][1]
        kp[0][1] = kp_mp[11][2]
        kp[1][0] = kp_mp[13][1]
        kp[1][1] = kp_mp[13][2]
        kp[2][0] = kp_mp[15][1]
        kp[2][1] = kp_mp[15][2]
        kp[3][0] = kp_mp[12][1]
        kp[3][1] = kp_mp[12][2]
        kp[4][0] = kp_mp[14][1]
        kp[4][1] = kp_mp[14][2]
        kp[5][0] = kp_mp[16][1]
        kp[5][1] = kp_mp[16][2]
        kp[6][0] = kp_mp[23][1]
        kp[6][1] = kp_mp[23][2]
        kp[7][0] = kp_mp[25][1]
        kp[7][1] = kp_mp[25][2]
        kp[8][0] = kp_mp[27][1]
        kp[8][1] = kp_mp[27][2]
        kp[9][0] = kp_mp[31][1]
        kp[9][1] = kp_mp[31][2]
        kp[10][0] = kp_mp[24][1]
        kp[10][1] = kp_mp[24][2]
        kp[11][0] = kp_mp[26][1]
        kp[11][1] = kp_mp[26][2]
        kp[12][0] = kp_mp[28][1]
        kp[12][1] = kp_mp[28][2]
        kp[13][0] = kp_mp[32][1]
        kp[13][1] = kp_mp[32][2]
    return kp


def important_kp_sens(kp_sens, count):
    '''
    Sort keypoints of sensors in given order (left arm, right arm, left leg, right leg) and safes into numpay array
    :param kp_mp: keypoints of open pose
    :return: kp: sorted list
    '''
    kp = np.zeros((12, 3))
    if kp_sens.any():
        kp[0][0] = kp_sens[count - 1][36]
        kp[0][1] = kp_sens[count - 1][37]
        kp[0][2] = kp_sens[count - 1][38]
        kp[1][0] = kp_sens[count - 1][39]
        kp[1][1] = kp_sens[count - 1][40]
        kp[1][2] = kp_sens[count - 1][41]
        kp[2][0] = kp_sens[count - 1][42]
        kp[2][1] = kp_sens[count - 1][43]
        kp[2][2] = kp_sens[count - 1][44]
        kp[3][0] = kp_sens[count - 1][24]
        kp[3][1] = kp_sens[count - 1][25]
        kp[3][2] = kp_sens[count - 1][26]
        kp[4][0] = kp_sens[count - 1][27]
        kp[4][1] = kp_sens[count - 1][28]
        kp[4][2] = kp_sens[count - 1][29]
        kp[5][0] = kp_sens[count - 1][30]
        kp[5][1] = kp_sens[count - 1][31]
        kp[5][2] = kp_sens[count - 1][32]
        kp[6][0] = kp_sens[count - 1][57]
        kp[6][1] = kp_sens[count - 1][58]
        kp[6][2] = kp_sens[count - 1][59]
        kp[7][0] = kp_sens[count - 1][60]
        kp[7][1] = kp_sens[count - 1][61]
        kp[7][2] = kp_sens[count - 1][62]
        kp[8][0] = kp_sens[count - 1][63]
        kp[8][1] = kp_sens[count - 1][64]
        kp[8][2] = kp_sens[count - 1][65]
        kp[9][0] = kp_sens[count - 1][45]
        kp[9][1] = kp_sens[count - 1][46]
        kp[9][2] = kp_sens[count - 1][47]
        kp[10][0] = kp_sens[count - 1][48]
        kp[10][1] = kp_sens[count - 1][49]
        kp[10][2] = kp_sens[count - 1][50]
        kp[11][0] = kp_sens[count - 1][51]
        kp[11][1] = kp_sens[count - 1][52]
        kp[11][2] = kp_sens[count - 1][53]

        # img = cv2.imread('Data/black.jpg')
        #
        # x = 500
        # x1 = 300
        # y = 500
        # cv2.circle(img, (x1, y), 7, (255, 0, 0), cv2.FILLED)  # Origin
        # # print(kp)
        # for i in range(len(kp)):
        #     cv2.circle(img, (int(kp[i][1] * x + x1), int(abs(kp[i][2]) * y)), 4, (0, 0, 255),
        #                cv2.FILLED)
        #
        # flippedImg = cv2.flip(img, 0)
        #
        # new_img = scale_img(flippedImg)
        # cv2.imshow('asdf', new_img)


        # for i in range(7):
        #     cv2.circle(img, (int(kp[16 + i][1] * x + x1), int(kp[16 + i][2] * y)), 4,
        #                (0, 255, 0), cv2.FILLED)
    return kp


def normalize_plane(frame, kp_op, kp_mp):
    '''
    Normalize the keypoints with a given frame
    :param frame: actual frame
    :param kp_op: important keypoints of openpose
    :param kp_mp: important keypoints of mediapipe
    :return: norm_kp_op: normalized keypoints of openpose, norm_kp_mp: normalized keypoints of mediapipe
    '''
    height, width = frame.shape[:2]
    norm_kp_op = np.zeros((14, 2))
    norm_kp_mp = np.zeros((14, 2))
    norm_kp_op[:, 0] = kp_op[:, 0] / width
    norm_kp_op[:, 1] = kp_op[:, 1] / height
    norm_kp_mp[:, 0] = kp_mp[:, 0] / width
    norm_kp_mp[:, 1] = kp_mp[:, 1] / height

    return norm_kp_op, norm_kp_mp

def get_init_points(kp_op):
    '''
    Saves needed points in an array for 3D --> 2D transformation
    :param kp_op: Keypoints of OpenPose
    :return: init_points array of points
    '''
    init_points = np.zeros((12, 2))
    if kp_op.any():
        init_points[0][0] = kp_op[0][0]     # Left shoulder
        init_points[0][1] = kp_op[0][1]
        init_points[1][0] = kp_op[3][0]     # Right shoulder
        init_points[1][1] = kp_op[3][1]
        init_points[2][0] = kp_op[6][0]     # Left hip
        init_points[2][1] = kp_op[6][1]
        init_points[3][0] = kp_op[10][0]     # Right hip
        init_points[3][1] = kp_op[10][1]
        init_points[4][0] = kp_op[1][0]     # Left elbow
        init_points[4][1] = kp_op[1][1]
        init_points[5][0] = kp_op[4][0]     # Right elbow
        init_points[5][1] = kp_op[4][1]
        init_points[6][0] = kp_op[7][0]     # Left knee
        init_points[6][1] = kp_op[7][1]
        init_points[7][0] = kp_op[11][0]     # Right knee
        init_points[7][1] = kp_op[11][1]
        init_points[8][0] = kp_op[2][0]     # Left wrist
        init_points[8][1] = kp_op[2][1]
        init_points[9][0] = kp_op[5][0]     # Right wrist
        init_points[9][1] = kp_op[5][1]
        init_points[10][0] = kp_op[8][0]     # Left ankle
        init_points[10][1] = kp_op[8][1]
        init_points[11][0] = kp_op[12][0]     # Right ankle
        init_points[11][1] = kp_op[12][1]

        # img = cv2.imread('Data/black.jpg')
        #
        # print(init_points)
        # for i in range(len(init_points)):
        #     cv2.circle(img, (int(init_points[i][0]), int(abs(init_points[i][1]))), 4, (0, 0, 255),
        #                cv2.FILLED)
        #
        # flippedImg = cv2.flip(img, 0)
        #
        # new_img = scale_img(flippedImg)
        # cv2.imshow('asdf', new_img)
        # cv2.waitKey(0)
    return init_points