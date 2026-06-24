import mediapipe as mp
import cv2

class Detector_MP():

    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
#        self.pose = self.mp_pose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)
        self.pose = self.mp_pose.Pose()


    def findKeypoints(self, actual_image, draw=True):
        lmList = []

        frameRGB = cv2.cvtColor(actual_image, cv2.COLOR_BGR2RGB)
        self.keypoints = self.pose.process(frameRGB)

        # Funktion obendrüber gibt entweder None oder ne Liste zurück deswegen folgender shit
        try:
            a = len(self.keypoints.pose_landmarks.landmark)
            go = True
        except AttributeError:
            go = False

        # print(len(self.keypoints.pose_landmarks.landmark))
        if go:
            for id, lm in enumerate(self.keypoints.pose_landmarks.landmark):
                x = int(lm.x * actual_image.shape[1])  # needed to get exact pixel value, MP normalize the coordinates
                y = int(lm.y * actual_image.shape[0])
                lmList.append([id, x, y])

            if self.keypoints.pose_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(actual_image, self.keypoints.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    cv2.imshow("Mediapipe", actual_image)

        return lmList

def main():
    return 0

if __name__ == "__main__":
    main()

