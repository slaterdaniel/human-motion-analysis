import cv2
import os
import numpy as np
import mediapipe as mp

np.set_printoptions(threshold=np.inf, suppress=True, precision=3, linewidth=95)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.99
)

mediapipe_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]

video_str = "../assets/filtered_videos/boetest.mp4"
video = cv2.VideoCapture(video_str)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(f'../outputs/mp_testing_{os.path.splitext(os.path.basename(video_str))[0]}.mp4', fourcc, fps, (width, height))

for curr_frame in range(frame_count):
    ret, frame = video.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ====================================================================
# Mediapipe
# ====================================================================

    mediapipe_results = pose.process(image)

    if mediapipe_results.pose_landmarks:
        current_pose = mediapipe_results.pose_landmarks.landmark


        for pair in mp_pose.POSE_CONNECTIONS:
            color = (0, 0, 255) if pair[0] % 2 else (255, 0, 0)
            cv2.line(frame, (int(current_pose[pair[0]].x * width), int(current_pose[pair[0]].y * height)),
                     (int(current_pose[pair[1]].x * width), int(current_pose[pair[1]].y * height)), color, 4)
    out.write(frame)

out.release()
video.release()





