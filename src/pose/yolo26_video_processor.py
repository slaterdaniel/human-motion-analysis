import cv2
from ultralytics import YOLO
import numpy as np
import pose.Engine as Engine
import os

def get_data(show=False, user_video=None):
    yolo = YOLO("../assets/yolo26x-pose.pt")
    valid_landmarks = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    all_data = []
    all_raw_data = []

    videos = Engine.find_videos(user_video)

    for video in videos:
        Engine.apply_filters(video)
        cap = cv2.VideoCapture('../assets/current_video.mp4')

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if user_video:
            user_skeleton, user_overlay = Engine.init_user_videos(width, height, fps)

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

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        data = np.zeros((frame_count, 42))  # 42 features

        results = yolo.predict(
            source=video,
            save=True if user_video else False,
            project="../../outputs/videos",
            name="overlays",
            exist_ok=True,
            show=show,
            show_boxes=False,
            stream=True)
        if user_video:
            os.rename(f"../outputs/videos/overlays/{os.path.splitext(os.path.basename(video))[0]}.mp4",
                      "../outputs/videos/overlays/full_overlay.mp4")

        for curr_frame, result in enumerate(results):
            current_pose = result.keypoints.xy[0].clone()

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
                    x = (lm[0] - center_x) / torso_length
                    y = (lm[1] - center_y) / torso_length

                    data[curr_frame, 16 + count * 2] = x
                    data[curr_frame, 17 + count * 2] = y
                    count += 1

            if user_video:
                # Initialize blank canvas
                canvas = np.zeros((height, width, 3), dtype=np.uint8)

                # Draw user skeleton on blank canvas
                for i, (pt1, pt2) in enumerate(connections):
                    lm1, lm2 = current_pose[pt1], current_pose[pt2]

                    lm1 = int(lm1[0] - center_x + (width / 2)), int(lm1[1] - center_y + (height / 2))
                    lm2 = int(lm2[0] - center_x + (width / 2)), int(lm2[1] - center_y + (height / 2))

                    cv2.line(canvas, lm1, lm2, (255, 255, 255), 2)

                user_skeleton.write(canvas)

            print(f"Frame: {curr_frame + 1}/{frame_count} Saved")

        cap.release()
        if user_video:
            user_skeleton.release()

        # smooth angle data to reduce noise
        smooth_window = 3
        half_window = smooth_window // 2
        for i in range(half_window, len(data) - half_window):
            data[i, :8] = np.mean(data[i - half_window:i + half_window + 1, :8], axis=0)

        # find angular velocities from smoothed angles
        for i in range(1, len(data)):
            data[i, 8:16] = data[i, :8] - data[i - 1, :8]

        window_size, border, step = Engine.get_formatting()
        all_raw_data.append(data[border: -border - 1])

        inputs = []
        for i in range(0, len(data) - window_size, step):
            window = data[i:i + window_size]
            inputs.append(window)

        inputs = np.array(inputs)
        print(inputs.shape)

        all_data.append(inputs)

    all_data = np.concatenate(all_data, axis=0)
    all_raw_data = np.concatenate(all_raw_data, axis=0)

    return all_data, all_raw_data
