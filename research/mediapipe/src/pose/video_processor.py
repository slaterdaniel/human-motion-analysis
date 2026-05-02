import cv2
import mediapipe as mp
import numpy as np
import os
import ffmpeg

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

    valid_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]
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
        ( # Convert all user_input to .mp4 with the same resolution
            ffmpeg
            .input(video)
            .filter('scale', width=1920, height=1080, force_original_aspect_ratio='decrease')
            .filter('pad', width=1920, height=1080, x='(ow-iw)/2', y='(oh-ih)/2')
            .output('../assets/current_video.mp4')
            .run(overwrite_output=True)
        )
        cap = cv2.VideoCapture('../assets/current_video.mp4')

        if user_video:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                "../outputs/videos/user_skeleton/user_skeleton.mp4",
                fourcc,
                fps,
                (width, height)
            )

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        data = np.zeros((frame_count, 50)) # 50 features

        curr_frame = 0
        prev_pose = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                current_pose = results.pose_landmarks.landmark

                for i, lm in enumerate(current_pose):
                    if prev_pose:
                        alpha = 0.5
                        prev_lm = prev_pose[i]
                        lm.x = alpha * prev_lm.x + (1 - alpha) * lm.x
                        lm.y = alpha * prev_lm.y + (1 - alpha) * lm.y

                # right shoulder angle
                a = (current_pose[14].x, current_pose[14].y)
                b = (current_pose[12].x, current_pose[12].y)
                c = (current_pose[24].x, current_pose[24].y)
                data[curr_frame, 0] = find_angle(a,b,c)

                # left shoulder angle
                a = (current_pose[13].x, current_pose[13].y)
                b = (current_pose[11].x, current_pose[11].y)
                c = (current_pose[23].x, current_pose[23].y)
                data[curr_frame, 1] = find_angle(a,b,c)

                # right elbow angle
                a = (current_pose[12].x, current_pose[12].y)
                b = (current_pose[14].x, current_pose[14].y)
                c = (current_pose[16].x, current_pose[16].y)
                data[curr_frame, 2] = find_angle(a,b,c)

                # left elbow angle
                a = (current_pose[11].x, current_pose[11].y)
                b = (current_pose[13].x, current_pose[13].y)
                c = (current_pose[15].x, current_pose[15].y)
                data[curr_frame, 3] = find_angle(a,b,c)

                # right hip angle
                a = (current_pose[12].x, current_pose[12].y)
                b = (current_pose[24].x, current_pose[24].y)
                c = (current_pose[26].x, current_pose[26].y)
                data[curr_frame, 4] = find_angle(a,b,c)

                # left hip angle
                a = (current_pose[11].x, current_pose[11].y)
                b = (current_pose[23].x, current_pose[23].y)
                c = (current_pose[25].x, current_pose[25].y)
                data[curr_frame, 5] = find_angle(a,b,c)

                # right knee angle
                a = (current_pose[24].x, current_pose[24].y)
                b = (current_pose[26].x, current_pose[26].y)
                c = (current_pose[28].x, current_pose[28].y)
                data[curr_frame, 6] = find_angle(a,b,c)

                # left knee angle
                a = (current_pose[23].x, current_pose[23].y)
                b = (current_pose[25].x, current_pose[25].y)
                c = (current_pose[27].x, current_pose[27].y)
                data[curr_frame, 7] = find_angle(a,b,c)

                # right ankle angle
                a = (current_pose[26].x, current_pose[26].y)
                b = (current_pose[28].x, current_pose[28].y)
                c = (current_pose[32].x, current_pose[32].y)
                data[curr_frame, 8] = find_angle(a,b,c)

                # left ankle angle
                a = (current_pose[25].x, current_pose[25].y)
                b = (current_pose[27].x, current_pose[27].y)
                c = (current_pose[31].x, current_pose[31].y)
                data[curr_frame, 9] = find_angle(a,b,c)

                # center coordinates around waist to normalize data across user_input
                center_x = (current_pose[23].x + current_pose[24].x) / 2
                center_y = (current_pose[23].y + current_pose[24].y) / 2

                # scale by torso length so different size people can be compared
                right_torso = float(np.linalg.norm(
                    np.array([current_pose[12].x, current_pose[12].y]) -
                    np.array([current_pose[24].x, current_pose[24].y])))

                left_torso = float(np.linalg.norm(
                    np.array([current_pose[11].x, current_pose[11].y]) -
                    np.array([current_pose[23].x, current_pose[23].y])))

                torso_length = (right_torso + left_torso) / 2
                count = 0

                for i, lm in enumerate(current_pose):
                    if i in valid_landmarks:
                        x = (lm.x - center_x) / torso_length
                        y = (lm.y - center_y) / torso_length

                        data[curr_frame, 20 + count*2] = x
                        data[curr_frame, 21 + count*2] = y
                        count += 1

                if user_video:
                    canvas = np.zeros((height, width, 3), dtype=np.uint8)
                    for connection in mp_pose.POSE_CONNECTIONS:
                        start = connection[0]
                        end = connection[1]

                        lm1 = results.pose_landmarks.landmark[start]
                        lm2 = results.pose_landmarks.landmark[end]

                        x1 = (lm1.x - center_x) / torso_length
                        y1 = (lm1.y - center_y) / torso_length

                        x1 = int((x1 / 3 + 0.5) * width)
                        y1 = int((y1 / 3 + 0.5) * height)

                        x2 = (lm2.x - center_x) / torso_length
                        y2 = (lm2.y - center_y) / torso_length

                        x2 = int((x2 / 3 + 0.5) * width)
                        y2 = int((y2 / 3 + 0.5) * height)

                        cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    out.write(canvas)

                prev_pose = [lm for lm in current_pose]

            curr_frame += 1

        cap.release()
        if user_video:
            out.release()
            print('\n\nskeleton saved\n\n')


        # smooth angle data to reduce noise
        smooth_window = 3
        half_window = smooth_window // 2
        for i in range(half_window, len(data) - half_window):
            data[i, :10] = np.mean(data[i - half_window:i + half_window + 1, :10], axis=0)

        # find angular velocities from smoothed angles
        for i in range(1, len(data)):
            data[i, 10:20] = data[i, :10] - data[i-1, :10]


        all_raw_data.append(data[border: -border-1])


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

