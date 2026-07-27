from ultralytics import YOLO
import cv2
import numpy as np
import time

def capture_pose(model: YOLO, frame: np.ndarray, yolo_landmarks):
   results = model.track(frame, persist=True, verbose=False, show=False, show_boxes=False, save=False)[0]
   speed = results.speed['inference']

   if len(results.keypoints.conf):
       keypoints = results.keypoints.xy[0].numpy()
       confidences = results.keypoints.conf[0].numpy()[yolo_landmarks]
       return keypoints, confidences, speed

   return None

def main():
   np.set_printoptions(threshold=np.inf, suppress=True, precision=3, linewidth=125)
   yolo_landmarks = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

   connections = [
       # Face
       (0, 1), (0, 2), (1, 3), (2, 4),
       # Shoulders and Arms
       (5, 6),  # Shoulder to Shoulder
       (5, 7), (7, 9),  # Left Arm (Shoulder-Elbow-Wrist)
       (6, 8), (8, 10),  # Right Arm (Shoulder-Elbow-Wrist)
       # Torso
       (5, 11), (6, 12),  # Shoulder to Hip (Left and Right)
       (11, 12),  # Hip to Hip
       # Legs (The core of your gait analysis)
       (11, 13), (13, 15),  # Left Leg (Hip-Knee-Ankle)
       (12, 14), (14, 16)  # Right Leg (Hip-Knee-Ankle)
   ]

   yolo_n = YOLO("../assets/yolo26_models/yolo26n-pose.pt", task='pose')
   yolo_x = YOLO("../assets/yolo26_models/yolo26x-pose.pt", task='pose')
   yolo_x.fuse()
   yolo_n.fuse()

   video_str = "../assets/filtered_videos/SHU-vsr-bounce-6.6mph.mp4"
   cap = cv2.VideoCapture(video_str)

   # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   # fps = int(cap.get(cv2.CAP_PROP_FPS))
   # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   #
   # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
   # out = cv2.VideoWriter('../outputs/yolotesting.mp4', fourcc, fps, (width, height))

   MIN_CONFIDENCE = 0.85

   s_count = 0
   x_count = 0
   start = time.time()

   nano_results = yolo_n.track(video_str, persist=True)
   nano_confs = []
   for i in range(len(nano_results)):
       nano_confs.append(nano_results[i].keypoints.conf[0].numpy()[yolo_landmarks])
   nano_confs = np.array(nano_confs)
   low_conf_args = set(np.argwhere(nano_confs < MIN_CONFIDENCE)[:, 0])

   print(low_conf_args)

   for low_conf_frame in low_conf_args:
       cap.set(cv2.CAP_PROP_POS_FRAMES, low_conf_frame)
       ret, frame = cap.read()
       keypoints, confidences, speed = capture_pose(model=yolo_x, frame=frame, yolo_landmarks=yolo_landmarks)

       # for pt1, pt2 in connections:
       #     cv2.line(frame, keypoints[pt1].astype(int), keypoints[pt2].astype(int), (255,255,255), 8)
       #
       # cv2.putText(frame, f'{model}: {np.min(confidences):.3f}', (int(width * 0.1), int(height * 0.1)),
       #             cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 6, cv2.LINE_AA)

       # out.write(frame)
       print(low_conf_frame, confidences, speed)

   cap.release()

main()
