import cv2
from ultralytics import YOLO
import numpy as np
import os
import ffmpeg

yolo = YOLO("../assets/yolo26s-pose.pt")

def find_angle(a, b, c):
    """
    find angle ∠ABC
    returned as float
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ac = a - b
    bc = c - b

    angle_radians = np.arccos(np.clip(np.dot(ac, bc) / (np.linalg.norm(ac) * np.linalg.norm(bc)), -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def get_data(user_video=None):
    valid_landmarks = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    window_size = 9
    border = window_size // 2
    step = 1

    all_data = []
    all_raw_data = []

    videos = []
    if user_video:
        videos = [user_video]
    else:
        video_folder = "../data/videos/training"
        for filename in os.listdir(video_folder):
            if filename == ".gitkeep": continue
            full_path = os.path.join(video_folder, filename)
            if os.path.isfile(full_path):
                videos.append(full_path)

        videos.pop()

    for video in videos:
        (  # Convert all user_input to .mp4 with the same resolution
            ffmpeg
            .input(video)
            .filter('scale', width=1920, height=1080, force_original_aspect_ratio='decrease')
            .filter('pad', width=1920, height=1080, x='(ow-iw)/2', y='(oh-ih)/2')
            .output('../assets/current_video.mp4')
            .run(overwrite_output=True)
        )
        cap = cv2.VideoCapture(video)
        width = 1920
        height = 1080

        if user_video:
            fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                "../assets/user_skeleton.mp4",
                fourcc,
                fps,
                (width, height)
            )

            STANDARD_SKELETON_CONNECTIONS = [
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
        data = np.zeros((frame_count, 42))  # 50 features

        results = yolo.predict(source=video, save=True if user_video else False, show=False)

        for curr_frame, result in enumerate(results):
            current_pose = result.keypoints.xy[0]

            # right shoulder angle
            a = (current_pose[8, 0], current_pose[8, 1])
            b = (current_pose[6, 0], current_pose[6, 1])
            c = (current_pose[12, 0], current_pose[12, 1])
            data[curr_frame, 0] = find_angle(a, b, c)

            # left shoulder angle
            a = (current_pose[7, 0], current_pose[7, 1])
            b = (current_pose[5, 0], current_pose[5, 1])
            c = (current_pose[11, 0], current_pose[11, 1])
            data[curr_frame, 1] = find_angle(a, b, c)

            # right elbow angle
            a = (current_pose[6, 0], current_pose[6, 1])
            b = (current_pose[8, 0], current_pose[8, 1])
            c = (current_pose[10, 0], current_pose[10, 1])
            data[curr_frame, 2] = find_angle(a, b, c)

            # left elbow angle
            a = (current_pose[5, 0], current_pose[5, 1])
            b = (current_pose[7, 0], current_pose[7, 1])
            c = (current_pose[9, 0], current_pose[9, 1])
            data[curr_frame, 3] = find_angle(a, b, c)

            # right hip angle
            a = (current_pose[6, 0], current_pose[6, 1])
            b = (current_pose[12, 0], current_pose[12, 1])
            c = (current_pose[14, 0], current_pose[14, 1])
            data[curr_frame, 4] = find_angle(a, b, c)

            # left hip angle
            a = (current_pose[5, 0], current_pose[5, 1])
            b = (current_pose[11, 0], current_pose[11, 1])
            c = (current_pose[13, 0], current_pose[13, 1])
            data[curr_frame, 5] = find_angle(a, b, c)

            # right knee angle
            a = (current_pose[12, 0], current_pose[12, 1])
            b = (current_pose[14, 0], current_pose[14, 1])
            c = (current_pose[16, 0], current_pose[16, 1])
            data[curr_frame, 6] = find_angle(a, b, c)

            # left knee angle
            a = (current_pose[11, 0], current_pose[11, 1])
            b = (current_pose[13, 0], current_pose[13, 1])
            c = (current_pose[15, 0], current_pose[15, 1])
            data[curr_frame, 7] = find_angle(a, b, c)

            # # right ankle angle
            # a = (current_pose[26, 0], current_pose[26, 1])
            # b = (current_pose[28, 0], current_pose[28, 1])
            # c = (current_pose[32, 0], current_pose[32, 1])
            # data[curr_frame, 8] = find_angle(a,b,c)
            #
            # # left ankle angle
            # a = (current_pose[25, 0], current_pose[25, 1])
            # b = (current_pose[27, 0], current_pose[27, 1])
            # c = (current_pose[31, 0], current_pose[31, 1])
            # data[curr_frame, 9] = find_angle(a,b,c)

            # center coordinates around waist to normalize data across user_input
            center_x = (current_pose[11, 0] + current_pose[12, 0]) / 2
            center_y = (current_pose[11, 1] + current_pose[12, 1]) / 2

            # scale by torso length so different size people can be compared
            right_torso = float(np.linalg.norm(
                np.array([current_pose[6, 0], current_pose[6, 1]]) -
                np.array([current_pose[12, 0], current_pose[12, 1]])))

            left_torso = float(np.linalg.norm(
                np.array([current_pose[5, 0], current_pose[5, 1]]) -
                np.array([current_pose[11, 0], current_pose[11, 1]])))

            torso_length = (right_torso + left_torso) / 2
            count = 0

            for i, lm in enumerate(current_pose):
                if i in valid_landmarks:
                    x = (lm[0] - center_x) * (height / 4) / torso_length
                    y = (lm[1] - center_y) * (height / 4) / torso_length

                    data[curr_frame, 16 + count * 2] = x
                    data[curr_frame, 17 + count * 2] = y
                    count += 1

            if user_video:
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
                for lm1, lm2 in STANDARD_SKELETON_CONNECTIONS:
                    x1 = int(current_pose[lm1, 0])
                    y1 = int(current_pose[lm1, 1])
                    x2 = int(current_pose[lm2, 0])
                    y2 = int(current_pose[lm2, 1])

                    x1 = int((x1 - center_x) * (height / 4) / torso_length)
                    y1 = int((y1 - center_y) * (height / 4) / torso_length)
                    x1 = int(x1 + (width / 2))
                    y1 = int(y1 + (height / 2))

                    x2 = int((x2 - center_x) * (height / 4) / torso_length)
                    y2 = int((y2 - center_y) * (height / 4) / torso_length)
                    x2 = int(x2 + (width / 2))
                    y2 = int(y2 + (height / 2))

                    cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
                out.write(canvas)

        cap.release()
        if user_video: out.release()

        # smooth angle data to reduce noise
        smooth_window = 3
        half_window = smooth_window // 2
        for i in range(half_window, len(data) - half_window):
            data[i, :8] = np.mean(data[i - half_window:i + half_window + 1, :8], axis=0)

        # find angular velocities from smoothed angles
        for i in range(1, len(data)):
            data[i, 8:16] = data[i, :8] - data[i - 1, :8]

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
    print(all_raw_data[0])

    return all_data, all_raw_data
