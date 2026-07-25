import cv2
import numpy as np
import os
import mediapipe as mp
from ultralytics import YOLO

import sys
from unittest.mock import MagicMock

mock_ext = MagicMock()
mock_ext.__spec__ = MagicMock()
sys.modules['mmcv._ext'] = mock_ext

os.environ['TORCH_HOME'] = 'mmpose'

from mmpose.apis import inference_topdown, init_model

np.set_printoptions(threshold=np.inf, suppress=True, precision=3, linewidth=95)

config_file = "mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py"
checkpoint_file = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
model = init_model(config_file, checkpoint_file, device='cpu')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=False,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
yolo = YOLO("assets/yolo26_models/yolo26x-pose.pt")
yolo.fuse()

mediapipe_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]
yolo_landmarks = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
mmpose_landmarks = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 21]

min_expected = 1.5
max_expected = 5

video_str = "data/user_input/short-boetest.mov"
video = cv2.VideoCapture(video_str)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

confidences = np.zeros((frame_count, 3, len(mediapipe_landmarks)))

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
        for i, lm in enumerate(mediapipe_landmarks):
            confidences[curr_frame, 0, i] = current_pose[lm].visibility

# ====================================================================
# YOLO26
# ====================================================================

    yolo_results = yolo.track(source=frame, persist=True, verbose=False)
    conf_scores = yolo_results[0].keypoints.conf[0]

    for i, lm in enumerate(yolo_landmarks):
        confidences[curr_frame, 1, i] = conf_scores[lm]

# ====================================================================
# MMPose
# ====================================================================

    mmpose_results = inference_topdown(model, frame)
    raw_scores = mmpose_results[0].pred_instances.keypoint_scores[0].astype(np.float64)

    # 2. Linearly scale between 0.0 and 1.0, then clip boundaries tightly
    scaled_scores = (raw_scores - min_expected) / (max_expected - min_expected)
    conf_scores = np.clip(scaled_scores, 0.0, 1.0)

    for i, lm in enumerate(mmpose_landmarks):
        confidences[curr_frame, 2, i] = conf_scores[lm]


video.release()

print(confidences)




