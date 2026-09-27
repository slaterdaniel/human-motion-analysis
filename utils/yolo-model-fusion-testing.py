from ultralytics import YOLO
from src.pose import Engine
import cv2
import numpy as np
import time

def capture_pose(model: YOLO, frame: np.ndarray, yolo_landmarks):
   results = model.track(frame, persist=True, verbose=False, show=False, show_boxes=False, save=False)[0]
   speed = results.speed['inference']

   if len(results.keypoints.conf):
       keypoints = results.keypoints.xy[0].numpy()[yolo_landmarks]
       confidences = results.keypoints.conf[0].numpy()[yolo_landmarks]
       return keypoints, confidences, speed

   return None

def fusion():
   np.set_printoptions(threshold=np.inf, suppress=True, precision=3, linewidth=125)
   yolo_landmarks = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

   connections = (
       (1, 7),  # left shoulder, hip
       (2, 8),  # right shoulder, hip
       (1, 3),  # left shoulder, elbow
       (2, 4),  # right shoulder, elbow
       (3, 5),  # left elbow, wrist
       (4, 6),  # right elbow, wrist
       (7, 9),  # left hip, knee
       (8, 10),  # right hip, knee
       (9, 11),  # left knee, ankle
       (10, 12),  # right knee, ankle
   )

   yolo_s = YOLO("../assets/yolo26_models/yolo26s-pose.pt", task='pose')
   yolo_x = YOLO("../assets/yolo26_models/yolo26x-pose.pt", task='pose')
   yolo_x.fuse()
   yolo_s.fuse()

   video_str = "../assets/filtered_videos/SHU-vf-normal-6.6mph.mp4"
   cap = cv2.VideoCapture(video_str)

   frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   fps = int(cap.get(cv2.CAP_PROP_FPS))
   width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

   fourcc = cv2.VideoWriter_fourcc(*"mp4v")
   out = cv2.VideoWriter('../outputs/yolotesting.mp4', fourcc, fps, (width, height))

   MIN_CONFIDENCE = 0.85

   s_count = 0
   x_count = 0
   start = time.time()

   for curr_frame in range(frame_count):
       ret, frame = cap.read()
       keypoints, confidences, speed = capture_pose(model=yolo_x, frame=frame, yolo_landmarks=yolo_landmarks)
       model = 'yolo_s'

       # if np.min(confidences) < MIN_CONFIDENCE:
       #     keypoints, confidences, speed = capture_pose(model=yolo_x, frame=frame, yolo_landmarks=yolo_landmarks)
       #     model = 'yolo_x'
       #     x_count += 1
       # else:
       #     s_count += 1

       for i, (pt1, pt2) in enumerate(connections):
           color = (0, 0, 255) if i % 2 else (0, 255, 0)
           cv2.line(frame, keypoints[pt1].astype(int), keypoints[pt2].astype(int), color, 4)

       # cv2.putText(frame, f'{model}: {np.min(confidences):.3f}', (int(width * 0.1), int(height * 0.1)),
       #             cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 6, cv2.LINE_AA)

       out.write(frame)
       print(model, confidences, speed)
   cap.release()
   out.release()

   print()
   print('SMALL:', s_count)
   print('XLARGE:', x_count)
   print('TIME:', time.time() - start)




def main():
   np.set_printoptions(threshold=np.inf, suppress=True, precision=3, linewidth=125)
   yolo_landmarks = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

   yolo_n = YOLO("../assets/yolo26_models/yolo26l-pose.pt", task='pose')
   yolo_x = YOLO("../assets/yolo26_models/yolo26x-pose.pt", task='pose')
   yolo_n.fuse()
   yolo_x.fuse()

   video_str = "../assets/filtered_videos/SHU-vsr-bounce-6.6mph.mp4"
   cap = cv2.VideoCapture(video_str)

   MIN_CONFIDENCE = 0.85

   start = time.time()

   small_results = yolo_n.track(video_str, persist=True, verbose=False)
   small_confs = []
   small_keypoints = []
   for i in range(len(small_results)):
      small_confs.append(small_results[i].keypoints.conf[0].numpy()[yolo_landmarks])
      small_keypoints.append(small_results[i].keypoints.xy[0].numpy()[yolo_landmarks])
   small_confs = np.array(small_confs)
   small_keypoints = np.array(small_keypoints)
   low_conf_args = set(np.argwhere(small_confs < MIN_CONFIDENCE)[:, 0])

   # print(low_conf_args)
   xl_results = []

   for low_conf_frame in low_conf_args:
      cap.set(cv2.CAP_PROP_POS_FRAMES, low_conf_frame)
      ret, frame = cap.read()
      keypoints, confidences, speed = capture_pose(model=yolo_x, frame=frame, yolo_landmarks=yolo_landmarks)
      xl_results.append(keypoints)

   print()
   print(f'TIME: {time.time() - start}')
   cap.release()

   # xl_results = np.array(xl_results)
   # final_results = small_keypoints.copy()
   # final_results[list(low_conf_args)] = xl_results
   # print(f'LENGTH: {len(low_conf_args)}')

   # print(final_results)

def adjust_limb_length(pair, keypoints, target_length):
    joint1 = keypoints[pair[0]]
    joint2 = keypoints[pair[1]]
    curr_length = np.linalg.norm(joint2 - joint1)
    original_line = joint2 - joint1

    ratio = target_length / curr_length
    new_line = (joint1.astype(int), (original_line * ratio + joint1).astype(int))

    return new_line

def limits():
    np.set_printoptions(threshold=np.inf, suppress=True, precision=3, linewidth=125)
    yolo_landmarks = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    yolo_s = YOLO("../assets/yolo26_models/yolo26s-pose.pt", task='pose')
    yolo_x = YOLO("../assets/yolo26_models/yolo26x-pose.pt", task='pose')
    yolo_s.fuse()
    yolo_x.fuse()

    keypoint_pairs = (
        (1, 7),  # left shoulder, hip
        (2, 8),  # right shoulder, hip
        (1, 3),  # left shoulder, elbow
        (2, 4),  # right shoulder, elbow
        (3, 5),  # left elbow, wrist
        (4, 6),  # right elbow, wrist
        (7, 9),   # left hip, knee
        (8, 10),  # right hip, knee
        (9, 11),  # left knee, ankle
        (10, 12), # right knee, ankle
    )

    video_str = "../data/user_input/SHU-vsr-bounce-6.6mph.mov"
    cap = cv2.VideoCapture(video_str)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter('../outputs/yolo_limits_testing.mp4', fourcc, fps, (width, height))

    MIN_CONFIDENCE = 0.85

    start = time.time()

    small_results = yolo_s.track(video_str, persist=True, verbose=False)
    small_confs = []
    small_keypoints = []
    for i in range(len(small_results)):
        small_confs.append(small_results[i].keypoints.conf[0].numpy()[yolo_landmarks])
        small_keypoints.append(small_results[i].keypoints.xy[0].numpy()[yolo_landmarks])
    # small_confs = np.array(small_confs)
    small_keypoints = np.array(small_keypoints)

    target_lengths = {x: np.linalg.norm(np.median(small_keypoints[:, x[0]] - small_keypoints[:, x[1]], axis=0))
                      for x in keypoint_pairs}

    for keypoints in small_keypoints:
        ret, frame = cap.read()

        for i, pair in enumerate(keypoint_pairs):
            new_line = adjust_limb_length(pair, keypoints, target_lengths[pair])

            keypoints[pair[0]] = new_line[0]
            keypoints[pair[1]] = new_line[1]
            color = (0, 0, 255) if i % 2 else (0, 255, 0)

            cv2.line(frame, new_line[0], new_line[1], color, 4)
            cv2.circle(frame, new_line[0], 5, color, -1)
            cv2.circle(frame, new_line[1], 5, color, -1)
        out.write(frame)

    out.release()

def smoothing():
    np.set_printoptions(threshold=np.inf, suppress=True, precision=3, linewidth=125)
    yolo_landmarks = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    yolo_s = YOLO("../assets/yolo26_models/yolo26s-pose.pt", task='pose')
    yolo_x = YOLO("../assets/yolo26_models/yolo26x-pose.pt", task='pose')
    yolo_s.fuse()
    yolo_x.fuse()

    keypoint_pairs = (
        (1, 7),  # left shoulder, hip
        (2, 8),  # right shoulder, hip
        (1, 3),  # left shoulder, elbow
        (2, 4),  # right shoulder, elbow
        (3, 5),  # left elbow, wrist
        (4, 6),  # right elbow, wrist
        (7, 9),   # left hip, knee
        (8, 10),  # right hip, knee
        (9, 11),  # left knee, ankle
        (10, 12), # right knee, ankle
    )

    video_str = "../data/user_input/SHU-vsr-bounce-6.6mph.mov"
    cap = cv2.VideoCapture(video_str)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter('../outputs/yolo_smoothing_testing.mp4', fourcc, fps, (width, height))

    MIN_CONFIDENCE = 0.85

    start = time.time()

    small_results = yolo_x.track(video_str, persist=True, verbose=False)
    small_keypoints = []
    for i in range(len(small_results)):
        small_keypoints.append(small_results[i].keypoints.xy[0].numpy()[yolo_landmarks])
    small_keypoints = np.array(small_keypoints)

    target_lengths = {x: np.linalg.norm(np.median(small_keypoints[:, x[0]] - small_keypoints[:, x[1]], axis=0))
                      for x in keypoint_pairs}

    for keypoints1, keypoints2, keypoints3, keypoints4 in zip(small_keypoints, small_keypoints[1:], small_keypoints[2:], small_keypoints[3:]):
        ret, frame = cap.read()

        for i, (kp1, kp2, kp3, kp4) in enumerate(zip(keypoints1.flatten(), keypoints2.flatten(), keypoints3.flatten(), keypoints4.flatten())):
            if ((kp1 < kp2 and kp3 < kp2 and kp4 > kp3) or (kp1 > kp2 and kp3 > kp2 and kp4 < kp3)) and abs(kp3 - kp2) > 30:
                print(kp1, ' ', kp2, ' ', kp3, ' ', kp4)
                cv2.putText(frame, f'MISTAKE: {i}', (int(width * 0.1), int(height * 0.1)),
                   cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 6, cv2.LINE_AA)

        for i, pair in enumerate(keypoint_pairs):
            new_line = adjust_limb_length(pair, keypoints1, target_lengths[pair])

            keypoints1[pair[0]] = new_line[0]
            keypoints1[pair[1]] = new_line[1]
            color = (0, 0, 255) if i % 2 else (0, 255, 0)

            cv2.line(frame, new_line[0], new_line[1], color, 4)
            cv2.circle(frame, new_line[0], 5, color, -1)
            cv2.circle(frame, new_line[1], 5, color, -1)

        out.write(frame)

    out.release()

fusion()