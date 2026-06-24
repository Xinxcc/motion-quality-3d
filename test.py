import numpy as np
import cv2
# a = [[1,1,1],[2,2,2]]
# b = [[3,3,3],[4,4,4]]
# c = [[5,5,5],[6,6,5]]
# d = [[7,7,7],[8,8,8]]
# e = np.array([a,b,c,d])
# f = e.reshape(-1,1,2)
# g = f.reshape(-1,2)
# h =np.hstack(e.squeeze()).T

# joints = [11, 13, 15, 12, 14, 16, 23, 25, 27, 24, 26, 28]
# a = 0
# aa = 2+joints[a]*4
# ab = 2+joints[a]*4+1
# ac = 2+joints[a]*4+2
# print(aa)

r = np.asarray([1.2178737734122753,1.2850929898072527,-1.0463995288874446])
(R, jac) = cv2.Rodrigues(r)
print(R)