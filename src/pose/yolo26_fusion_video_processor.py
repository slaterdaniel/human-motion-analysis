import cv2
from ultralytics import YOLO
import numpy as np
import src.pose.Engine as Engine
import os

def get_data(show=False, user_video=None):
    yolo_s = YOLO("assets/yolo26_models/yolo26s-pose.pt")
    yolo_x = YOLO("assets/yolo26_models/yolo26x-pose.pt")
    yolo_s.fuse()
    yolo_x.fuse()

    valid_landmarks = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    MIN_CONFIDENCE = 0.75

    all_data = []
    all_raw_data = []

    videos = Engine.find_videos(user_video)

    for video in videos:
        Engine.apply_filters(video)
        video_basename = os.path.splitext(os.path.basename(video))[0]
        filtered_path = f'assets/filtered_videos/{video_basename}.mp4'
        cap = cv2.VideoCapture(filtered_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if user_video:
            user_skeleton, user_overlay = Engine.init_user_videos(width, height, fps)
            feature_coords = np.zeros((frame_count, 46))  # 50 features

        connections = [
            # Head
            (0, 1), (0, 2), (1, 3), (2, 4),
            # Arms
            (5, 6),  # Shoulder to Shoulder
            (5, 7), (7, 9),  # Left Arm
            (6, 8), (8, 10),  # Right Arm
            # Torso
            (5, 11), (6, 12),  # Shoulder to Hip
            (11, 12),  # Hip to Hip
            # Legs
            (11, 13), (13, 15),  # Left Leg
            (12, 14), (14, 16)  # Right Leg
        ]

        data = np.zeros((frame_count, 46))  # 46 features

        # if user_video:
        #     os.rename(f"outputs/videos/overlays/{video_basename}.mp4",
        #               "outputs/videos/overlays/full_overlay.mp4")

        for curr_frame in range(frame_count):
            ret, frame = cap.read()
            result = yolo_s.track(
                source=frame,
                show=show,
                verbose=False,
                show_boxes=False)[0].keypoints

            lowest_confidence = np.min(result.conf[0].numpy()[valid_landmarks])
            if lowest_confidence < MIN_CONFIDENCE:
                result = yolo_x.track(
                    source=frame,
                    show=show,
                    verbose=False,
                    show_boxes=False)[0].keypoints

            current_pose = result.xy[0].clone()

            # right shoulder angle
            a = current_pose[8]
            b = current_pose[6]
            c = current_pose[12]
            data[curr_frame, 0] = Engine.find_angle(a, b, c)

            # left shoulder angle
            a = current_pose[7]
            b = current_pose[5]
            c = current_pose[11]
            data[curr_frame, 1] = Engine.find_angle(a, b, c)

            # right elbow angle
            a = current_pose[6]
            b = current_pose[8]
            c = current_pose[10]
            data[curr_frame, 2] = Engine.find_angle(a, b, c)

            # left elbow angle
            a = current_pose[5]
            b = current_pose[7]
            c = current_pose[9]
            data[curr_frame, 3] = Engine.find_angle(a, b, c)

            # right hip angle
            a = current_pose[6]
            b = current_pose[12]
            c = current_pose[14]
            data[curr_frame, 4] = Engine.find_angle(a, b, c)

            # left hip angle
            a = current_pose[5]
            b = current_pose[11]
            c = current_pose[13]
            data[curr_frame, 5] = Engine.find_angle(a, b, c)

            # right knee angle
            a = current_pose[12]
            b = current_pose[14]
            c = current_pose[16]
            data[curr_frame, 6] = Engine.find_angle(a, b, c)

            # left knee angle
            a = current_pose[11]
            b = current_pose[13]
            c = current_pose[15]
            data[curr_frame, 7] = Engine.find_angle(a, b, c)

            # center coordinates around waist to normalize data across user_input
            center_x = (current_pose[11, 0] + current_pose[12, 0]) / 2
            center_y = (current_pose[11, 1] + current_pose[12, 1]) / 2

            # scale by torso length so different size people can be compared
            right_torso = float(np.linalg.norm(current_pose[6] - current_pose[12]))
            left_torso = float(np.linalg.norm(current_pose[5] - current_pose[11]))
            torso_length = (right_torso + left_torso) / 2
            count = 0

            for i, lm in enumerate(current_pose):
                if i in valid_landmarks:
                    x = lm[0] - center_x
                    y = lm[1] - center_y

                    data[curr_frame, 20 + count * 2] = x / torso_length
                    data[curr_frame, 21 + count * 2] = y / torso_length

                    if user_video:
                        feature_coords[curr_frame, 20 + count * 2] = x + (width / 2)
                        feature_coords[curr_frame, 21 + count * 2] = y + (height / 2)

                    count += 1

            if user_video:
                # Initialize blank canvas
                canvas = np.zeros((height, width, 3), dtype=np.uint8)

                # Draw user skeleton on blank canvas
                drawn = set()
                for i, (pt1, pt2) in enumerate(connections):
                    lm1, lm2 = current_pose[pt1], current_pose[pt2]
                    cv2.line(frame, (int(lm1[0]), int(lm1[1])), (int(lm2[0]), int(lm2[1])), (255, 255, 255), 4)

                    lm1 = int(lm1[0] - center_x + (width / 2)), int(lm1[1] - center_y + (height / 2))
                    lm2 = int(lm2[0] - center_x + (width / 2)), int(lm2[1] - center_y + (height / 2))

                    cv2.line(canvas, lm1, lm2, (255, 255, 255), 5)
                    if lm1 not in drawn:
                        cv2.circle(canvas, (int(lm1[0]), int(lm1[1])), 8, (255, 255, 255), -1)
                        drawn.add(lm1)
                    if lm2 not in drawn:
                        cv2.circle(canvas, (int(lm2[0]), int(lm2[1])), 8, (255, 255, 255), -1)
                        drawn.add(lm2)

                user_skeleton.write(canvas)
                user_overlay.write(frame)

            print(f"Frame: {curr_frame + 1}/{frame_count} Saved")

        cap.release()
        if user_video:
            user_skeleton.release()
            user_overlay.release()

        # smooth angle data to reduce noise
        smooth_window = 3
        half_window = smooth_window // 2
        for i in range(half_window, len(data) - half_window):
            data[i, :8] = np.mean(data[i - half_window:i + half_window + 1, :8], axis=0)

        # find angular velocities from smoothed angles
        for i in range(1, len(data)):
            data[i, 10:18] = data[i, :8] - data[i - 1, :8]

        window_size, border, step = Engine.get_formatting()
        all_raw_data.append(data[border: -border - 1])

        if user_video:
            feature_coords[:, :18] = data[:, :18]

        inputs = []
        for i in range(0, len(data) - window_size, step):
            window = data[i:i + window_size]
            inputs.append(window)

        inputs = np.array(inputs)
        print(inputs.shape)

        all_data.append(inputs)

    all_data = np.concatenate(all_data, axis=0)
    all_raw_data = np.concatenate(all_raw_data, axis=0)

    if user_video:
        return all_data, all_raw_data, np.array(feature_coords).astype(int)

    return all_data, all_raw_data
