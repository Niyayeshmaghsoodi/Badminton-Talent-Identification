import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture("data/raw/videos/sample.mp4")
poses = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        keypoints = [(lm.x, lm.y) for lm in result.pose_landmarks.landmark]
        poses.append(keypoints)

cap.release()
np.save("data/processed/poses.npy", poses)
