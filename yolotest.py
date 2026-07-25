from ultralytics import YOLO
import cv2
import numpy as np

np.set_printoptions(threshold=np.inf, suppress=True, precision=3, linewidth=125)

def capture_pose(model: YOLO, frame: np.ndarray):
    yolo_landmarks = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    results = model.track(frame, persist=True, verbose=False, show=True, show_boxes=False, save=False)[0]
    time = results.speed['inference']

    if len(results.keypoints.conf):
        confidences = results.keypoints.conf[0].numpy()[yolo_landmarks]
        return confidences, time

    return None

yolo_n = YOLO("assets/yolo26_models/yolo26n-pose.pt", task='pose')
yolo_x = YOLO("assets/yolo26_models/yolo26x-pose.pt", task='pose')
yolo_n.fuse()
yolo_x.fuse()

video_str = "assets/filtered_videos/short-boetest.mp4"
cap = cv2.VideoCapture(video_str)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

MIN_CONFIDENCE = 0.5

for curr_frame in range(frame_count):
    ret, frame = cap.read()

    confidences, time = capture_pose(model=yolo_n, frame=frame)
    model = 'yolo_n'

    if np.min(confidences) < MIN_CONFIDENCE:
        confidences, time = capture_pose(model=yolo_x, frame=frame)
        model = 'yolo_x'

    print(model, confidences, time)

cap.release()