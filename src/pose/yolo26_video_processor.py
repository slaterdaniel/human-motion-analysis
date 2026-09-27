import cv2
from ultralytics import YOLO
import numpy as np
import src.pose.Engine as Engine
from src.pose.features import *
import os

def get_data(show=False, user_video=None):
    yolo = YOLO("assets/yolo26_models/yolo26x-pose.pt")
    yolo.fuse()

    valid_landmarks = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    NOSE_KP       = 0
    L_SHOULDER_KP = 5
    R_SHOULDER_KP = 6
    L_ELBOW_KP    = 7
    R_ELBOW_KP    = 8
    L_WRIST_KP    = 9
    R_WRIST_KP    = 10
    L_HIP_KP      = 11
    R_HIP_KP      = 12
    L_KNEE_KP     = 13
    R_KNEE_KP     = 14
    L_ANKLE_KP    = 15
    R_ANKLE_KP    = 16
    L_FOOT_KP     = None
    R_FOOT_KP     = None

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
        cap.release()

        if user_video:
            user_skeleton, user_overlay = Engine.init_user_videos(width, height, fps)
            user_overlay.release()
            feature_coords = np.zeros((frame_count, 62))  # 50 features

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
            # Legs
            (11, 13), (13, 15),  # Left Leg (Hip-Knee-Ankle)
            (12, 14), (14, 16)  # Right Leg (Hip-Knee-Ankle)
        ]

        data = np.zeros((frame_count, 62))  # 46 features

        results = yolo.track(
            source=filtered_path,
            persist=True,
            save=True if user_video else False,
            project="/Users/danielslater/Documents/human-motion-analysis/outputs/videos",
            name="overlays",
            exist_ok=True,
            show=show,
            show_boxes=False)

        if user_video:
            os.rename(f"outputs/videos/overlays/{video_basename}.mp4",
                      "outputs/videos/overlays/full_overlay.mp4")

        for curr_frame, result in enumerate(results):
            current_pose = result.keypoints.xy[0].clone()

            # right shoulder angle
            a = current_pose[R_ELBOW_KP]
            b = current_pose[R_SHOULDER_KP]
            c = current_pose[R_HIP_KP]
            data[curr_frame, R_SHOULDER_ANGLE] = Engine.find_angle(a, b, c)

            # left shoulder angle
            a = current_pose[L_ELBOW_KP]
            b = current_pose[L_SHOULDER_KP]
            c = current_pose[L_HIP_KP]
            data[curr_frame, L_SHOULDER_ANGLE] = Engine.find_angle(a, b, c)

            # right elbow angle
            a = current_pose[R_SHOULDER_KP]
            b = current_pose[R_ELBOW_KP]
            c = current_pose[R_WRIST_KP]
            data[curr_frame, R_ELBOW_ANGLE] = Engine.find_angle(a, b, c)

            # left elbow angle
            a = current_pose[L_SHOULDER_KP]
            b = current_pose[L_ELBOW_KP]
            c = current_pose[L_WRIST_KP]
            data[curr_frame, L_ELBOW_ANGLE] = Engine.find_angle(a, b, c)

            # right hip angle
            a = current_pose[R_SHOULDER_KP]
            b = current_pose[R_HIP_KP]
            c = current_pose[R_KNEE_KP]
            data[curr_frame, R_HIP_ANGLE] = Engine.find_angle(a, b, c)

            # left hip angle
            a = current_pose[L_SHOULDER_KP]
            b = current_pose[L_HIP_KP]
            c = current_pose[L_KNEE_KP]
            data[curr_frame, L_HIP_ANGLE] = Engine.find_angle(a, b, c)

            # right knee angle
            a = current_pose[R_HIP_KP]
            b = current_pose[R_KNEE_KP]
            c = current_pose[R_ANKLE_KP]
            data[curr_frame, R_KNEE_ANGLE] = Engine.find_angle(a, b, c)

            # left knee angle
            a = current_pose[L_HIP_KP]
            b = current_pose[L_KNEE_KP]
            c = current_pose[L_ANKLE_KP]
            data[curr_frame, L_KNEE_ANGLE] = Engine.find_angle(a, b, c)

            # right tibial angle
            a = current_pose[R_KNEE_KP]
            b = current_pose[R_ANKLE_KP]
            c = (current_pose[R_ANKLE_KP, 0], current_pose[R_ANKLE_KP, 1] - 100)
            data[curr_frame, R_TIBIAL_ANGLE] = Engine.find_angle(a, b, c)

            # left tibial angle
            a = current_pose[L_KNEE_KP]
            b = current_pose[L_ANKLE_KP]
            c = (current_pose[L_ANKLE_KP, 0], current_pose[L_ANKLE_KP, 1] - 100)
            data[curr_frame, L_TIBIAL_ANGLE] = Engine.find_angle(a, b, c)

            if R_FOOT_KP:
                # right foot inclination angle
                a = (current_pose[R_ANKLE_KP, 0] + 100, current_pose[R_ANKLE_KP, 1])
                b = current_pose[R_ANKLE_KP]
                c = current_pose[R_FOOT_KP]
                data[curr_frame, R_FOOT_INCLINATION_ANGLE] = Engine.find_angle(a, b, c)

                # left foot inclination angle
                a = (current_pose[L_ANKLE_KP, 0] + 100, current_pose[L_ANKLE_KP, 1])
                b = current_pose[L_ANKLE_KP]
                c = current_pose[L_FOOT_KP]
                data[curr_frame, L_FOOT_INCLINATION_ANGLE] = Engine.find_angle(a, b, c)

            # right tibial angle
            a = (current_pose[R_HIP_KP, 0] + 100, current_pose[R_HIP_KP, 1])
            b = current_pose[R_HIP_KP]
            c = current_pose[R_SHOULDER_KP]
            data[curr_frame, R_FORWARD_LEAN] = Engine.find_angle(a, b, c)

            # left forward lean angle
            a = (current_pose[L_HIP_KP, 0] + 100, current_pose[L_HIP_KP, 1])
            b = current_pose[L_HIP_KP]
            c = current_pose[L_SHOULDER_KP]
            data[curr_frame, L_FORWARD_LEAN] = Engine.find_angle(a, b, c)

            # center coordinates around waist to normalize data across user_input
            center_x = (current_pose[L_HIP_KP, 0] + current_pose[R_HIP_KP, 0]) / 2
            center_y = (current_pose[L_HIP_KP, 1] + current_pose[R_HIP_KP, 1]) / 2

            # scale by torso length so different size people can be compared
            right_torso = float(np.linalg.norm(current_pose[R_SHOULDER_KP] - current_pose[R_HIP_KP]))
            left_torso = float(np.linalg.norm(current_pose[L_SHOULDER_KP] - current_pose[L_HIP_KP]))
            torso_length = (right_torso + left_torso) / 2
            count = 0

            for i, lm in enumerate(current_pose):
                if i in valid_landmarks:
                    x = lm[0] - center_x
                    y = lm[1] - center_y

                    data[curr_frame, NOSE_X + count * 2] = x / torso_length
                    data[curr_frame, NOSE_Y + count * 2] = y / torso_length

                    if user_video:
                        feature_coords[curr_frame, NOSE_X + count * 2] = x + (width / 2)
                        feature_coords[curr_frame, NOSE_Y + count * 2] = y + (height / 2)

                    count += 1


            if user_video:
                # Initialize blank canvas
                canvas = np.zeros((height, width, 3), dtype=np.uint8)

                # Draw user skeleton on blank canvas
                drawn = set()
                for i, (pt1, pt2) in enumerate(connections):
                    lm1, lm2 = current_pose[pt1], current_pose[pt2]

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

            print(f"Frame: {curr_frame + 1}/{frame_count} Saved")

        if user_video:
            user_skeleton.release()

        # smooth angle data to reduce noise
        # smooth_window = 3
        # half_window = smooth_window // 2
        # for i in range(half_window, len(data) - half_window):
        #     data[i, :R_SHOULDER_ANGLE_VEL] = np.mean(data[i - half_window:i + half_window + 1, :R_SHOULDER_ANGLE_VEL], axis=0)

        # find angular velocities from smoothed angles
        for i in range(1, len(data)):
            data[i, R_SHOULDER_ANGLE_VEL:NOSE_X] = data[i, :R_SHOULDER_ANGLE_VEL] - data[i - 1, :R_SHOULDER_ANGLE_VEL]

        window_size, border, step = Engine.get_formatting()
        all_raw_data.append(data[border: -border - 1])

        if user_video:
            feature_coords[:, :NOSE_X] = data[:, :NOSE_X]

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
