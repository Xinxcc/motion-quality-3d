import cv2
import functions_general as fg
import numpy as np
from randommmm import randrange

def optimize_rot_trans(kp_3D, kp_2D, cam, dist):
    '''
    Computes rotation and translation vector for multiple points. Than computes mean of each element.
    :param kp_3D: Positions in space
    :param kp_2D: Positions in plane
    :param cam: Camera matrix
    :param dist: Distortion vector
    :return: optimized rotation and translation vector
    '''
    number_of_vectors = 10000

    rot = np.zeros((number_of_vectors, 3, 1))
    trans = np.zeros((number_of_vectors, 3, 1))
    kp_2D_temp = np.zeros((6, 2))
    kp_3D_temp = np.zeros((6, 3))
    length = kp_2D.shape[0]
    rand_vec = np.zeros((6, 1))
    for i in range(number_of_vectors):
        count = 0
        while True:                             # Getting random vector
            rand_numb = randrange(0, length)
            if rand_numb not in rand_vec:
                rand_vec[count] = rand_numb
                count += 1
            if count == 6:
                break

        for j in range(rand_vec.shape[0]):      # Getting random 2D and 3D points
            kp_2D_temp[j] = [kp_2D[int(rand_vec[j][0])][0], kp_2D[int(rand_vec[j][0])][1]]
            kp_3D_temp[j] = [kp_3D[int(rand_vec[j][0])][0], kp_3D[int(rand_vec[j][0])][1], kp_3D[int(rand_vec[j][0])][2]]

        # Computing rotation and translation vectors
        ret, r_vec, t_vec = cv2.solvePnP(kp_3D_temp, kp_2D_temp, cam, dist, flags=cv2.SOLVEPNP_EPNP )
        rot[i] = r_vec
        trans[i] = t_vec

    # np.squeeze(r_vec, axis=2)

    # Computing mean over all rotation and translation vectors
    r_vec[0][0] = np.mean(rot[:, 0, 0])
    r_vec[1][0] = np.mean(rot[:, 1, 0])
    r_vec[2][0] = np.mean(rot[:, 2, 0])
    t_vec[0][0] = np.mean(trans[:, 0, 0])
    t_vec[1][0] = np.mean(trans[:, 1, 0])
    t_vec[2][0] = np.mean(trans[:, 2, 0])

    return r_vec, t_vec


def get_rot_trans(kp_sens_3D, kp_2D, frame):

    # Setting up Cam-Matrix done like --> https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
    # Camera calibration should be done to have better results
    # size = frame.shape
    # focal_length = size[1]
    # center = (size[1]/2, size[0]/2)
    # cam_mtx = np.array([(focal_length, 0, center[0]),
    #                     (0, focal_length, center[1]),
    #                     (0, 0, 1)])
    #
    # dist = np.zeros((4, 1))

    # With camera calibration
    cam_mtx = np.loadtxt('Calibration/Parameters/Logitech_USB_Matrix_01.txt')
    dist = np.loadtxt('Calibration/Parameters/Logitech_USB_dist_01.txt')
    dist = np.reshape(dist, (-1, 1))

    kp_3D = np.array([(kp_sens_3D[0][0], kp_sens_3D[0][1], kp_sens_3D[0][2]),       # Left shoulder joint
             (kp_sens_3D[3][0], kp_sens_3D[3][1], kp_sens_3D[3][2]),                # Right shoulder joint
             (kp_sens_3D[6][0], kp_sens_3D[6][1], kp_sens_3D[6][2]),                # Left hip joint
             (kp_sens_3D[9][0], kp_sens_3D[9][1], kp_sens_3D[9][2]),                # Right hip joint
             (kp_sens_3D[1][0], kp_sens_3D[1][1], kp_sens_3D[1][2]),                # Left elbow joint
             (kp_sens_3D[4][0], kp_sens_3D[4][1], kp_sens_3D[4][2]),                # Right elbow joint
             (kp_sens_3D[7][0], kp_sens_3D[7][1], kp_sens_3D[7][2]),                # Left knee joint
             (kp_sens_3D[10][0], kp_sens_3D[10][1], kp_sens_3D[10][2]),             # Right knee joint
             (kp_sens_3D[2][0], kp_sens_3D[2][1], kp_sens_3D[2][2]),                # Left wrist joint
             (kp_sens_3D[5][0], kp_sens_3D[5][1], kp_sens_3D[5][2]),                # Right wrist joint
             (kp_sens_3D[8][0], kp_sens_3D[8][1], kp_sens_3D[8][2]),                # Left ankle joint
             (kp_sens_3D[11][0], kp_sens_3D[11][1], kp_sens_3D[11][2])])            # Right ankle joint
    # ])

    # kp_3D and kp_2D need to be filled with float-type numbers
    kp_2D = kp_2D.astype(np.float)

    if kp_2D.shape[0] > 6:
        r_vec, t_vec = optimize_rot_trans(kp_3D, kp_2D, cam_mtx, dist)
    else:
        ret, r_vec, t_vec = cv2.solvePnP(kp_3D, kp_2D, cam_mtx, dist)

    return r_vec, t_vec, cam_mtx, dist

def space_to_plane(kp_3D, rot, trans, cam, dist):
    kp_2D, jacobian = cv2.projectPoints(kp_3D, rot, trans, cam, dist)
    # print(kp_2D)
    return kp_2D
