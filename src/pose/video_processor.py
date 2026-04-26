import cv2
import numpy as np
import os
import ffmpeg
import time

class Engine:
    window_size = 9
    border = window_size // 2
    step = 1
    
    @staticmethod
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

    @staticmethod
    def find_videos(user_video=None):
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
            videos.pop()  # REMOVE LATER

        return videos

    @staticmethod
    def apply_filters(video):
        (
            ffmpeg
            .input(video)
            # 3. Boost contrast to help the AI see limbs against the treadmill
            .filter('eq', contrast=1.3, brightness=0.02)
            .output('../assets/current_video.mp4', pix_fmt='yuv420p', crf=18)
            .run(overwrite_output=True)
        )

    @staticmethod
    def init_user_videos(width, height, fps):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        user_skeleton = cv2.VideoWriter(
            "../outputs/videos/user_skeleton/user_skeleton.mp4",
            fourcc,
            fps,
            (width, height)
        )
        user_overlay = cv2.VideoWriter(
            "../outputs/videos/overlays/full_overlay.mp4",
            fourcc,
            fps,
            (width, height)
        )
        return user_skeleton, user_overlay


class MMPose(Engine):
    def __init__(self):
        import sys
        from unittest.mock import MagicMock
        mock_ext = MagicMock()
        mock_ext.__spec__ = MagicMock()
        sys.modules['mmcv._ext'] = mock_ext

        os.environ['TORCH_HOME'] = '../mmpose'

        from mmpose.apis import inference_topdown, init_model
        self.inference_topdown = inference_topdown
        config_file = "../mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py"
        checkpoint_file = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"

        self.model = init_model(config_file, checkpoint_file, device='cpu')
        self.connections = self.model.dataset_meta['skeleton_links'][:25]
        self.connection_colors = self.model.dataset_meta['skeleton_link_colors'][:25]


    def __str__(self):
        return "MMPOSE ENGINE"

    def get_data(self, user_video=None):

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

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            data = np.zeros((frame_count, 50))  # 50 features

            for curr_frame in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break

                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.inference_topdown(model, frame)  # Run the model

                # Extract data
                keypoints = results[0].pred_instances.keypoints[0].astype(np.float64)  # Array of [x, y]

                # right shoulder angle
                a = keypoints[8]
                b = keypoints[6]
                c = keypoints[12]
                data[curr_frame, 0] = Engine.find_angle(a, b, c)

                # left shoulder angle
                a = keypoints[7]
                b = keypoints[5]
                c = keypoints[11]
                data[curr_frame, 1] = Engine.find_angle(a, b, c)

                # right elbow angle
                a = keypoints[6]
                b = keypoints[8]
                c = keypoints[10]
                data[curr_frame, 2] = Engine.find_angle(a, b, c)

                # left elbow angle
                a = keypoints[5]
                b = keypoints[7]
                c = keypoints[9]
                data[curr_frame, 3] = Engine.find_angle(a, b, c)

                # right hip angle
                a = keypoints[6]
                b = keypoints[12]
                c = keypoints[14]
                data[curr_frame, 4] = Engine.find_angle(a, b, c)

                # left hip angle
                a = keypoints[5]
                b = keypoints[11]
                c = keypoints[13]
                data[curr_frame, 5] = Engine.find_angle(a, b, c)

                # right knee angle
                a = keypoints[12]
                b = keypoints[14]
                c = keypoints[16]
                data[curr_frame, 6] = Engine.find_angle(a, b, c)

                # left knee angle
                a = keypoints[11]
                b = keypoints[13]
                c = keypoints[15]
                data[curr_frame, 7] = Engine.find_angle(a, b, c)

                # right ankle angle
                a = keypoints[14]
                b = keypoints[16]
                c = keypoints[21]
                data[curr_frame, 8] = Engine.find_angle(a, b, c)

                # left ankle angle
                a = keypoints[13]
                b = keypoints[15]
                c = keypoints[18]
                data[curr_frame, 9] = Engine.find_angle(a, b, c)

                # center coordinates around waist to normalize data across user_input
                center_x = (keypoints[11, 0] + keypoints[12, 0]) / 2
                center_y = (keypoints[11, 1] + keypoints[12, 1]) / 2

                # scale by torso length so different size people can be compared
                right_torso = np.linalg.norm(keypoints[6] - keypoints[12])
                left_torso = np.linalg.norm(keypoints[5] - keypoints[11])

                torso_length = (right_torso + left_torso) / 2
                for i, lm in enumerate(np.vstack((keypoints[0], keypoints[5:17], keypoints[18], keypoints[21]))):
                    x = (lm[0] - center_x) / torso_length
                    y = (lm[1] - center_y) / torso_length

                    data[curr_frame, 20 + i * 2] = x
                    data[curr_frame, 21 + i * 2] = y

                if user_video:
                    # Initialize blank canvas
                    canvas = np.zeros((height, width, 3), dtype=np.uint8)

                    # Draw circles for hands
                    rhand = int(keypoints[9, 0] - center_x + (width / 2)), int(keypoints[9, 1] - center_y + (height / 2))
                    lhand = int(keypoints[10, 0] - center_x + (width / 2)), int(keypoints[10, 1] - center_y + (height / 2))
                    cv2.circle(canvas, rhand, 5, (255, 255, 255), -1)
                    cv2.circle(canvas, lhand, 5, (255, 255, 255), -1)

                cv2.circle(frame, keypoints[9].astype(int), 5, tuple(map(int, connection_colors[10])), -1)
                cv2.circle(frame, keypoints[10].astype(int), 5, tuple(map(int, connection_colors[11])), -1)

                # Draw skeleton connection on original video
                for i, (pt1, pt2) in enumerate(connections):
                    lm1, lm2 = keypoints[pt1].astype(int), keypoints[pt2].astype(int)
                    color = tuple(map(int, connection_colors[i]))
                    cv2.line(frame, lm1, lm2, color, 4)

                    # Draw user skeleton on blank canvas
                    if user_video:
                        lm1 = int(lm1[0] - center_x + (width / 2)), int(lm1[1] - center_y + (height / 2))
                        lm2 = int(lm2[0] - center_x + (width / 2)), int(lm2[1] - center_y + (height / 2))
                        cv2.line(canvas, lm1, lm2, (255, 255, 255), 2)

                if user_video:
                    user_skeleton.write(canvas)
                    user_overlay.write(frame)

                cv2.imshow('Processing Preview', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                print(f"Frame: {curr_frame + 1}/{frame_count} Saved")

            cap.release()
            if user_video:
                user_skeleton.release()
                user_overlay.release()
                cv2.destroyAllWindows()

            # smooth angle data to reduce noise
            smooth_window = 3
            half_window = smooth_window // 2
            for i in range(half_window, len(data) - half_window):
                data[i, :10] = np.mean(data[i - half_window:i + half_window + 1, :10], axis=0)

            # find angular velocities from smoothed angles
            for i in range(1, len(data)):
                data[i, 10:20] = data[i, :10] - data[i - 1, :10]

            all_raw_data.append(data[Engine.border: -Engine.border - 1])

            inputs = []
            for i in range(0, len(data) - Engine.window_size, Engine.step):
                window = data[i:i + Engine.window_size]
                inputs.append(window)

            inputs = np.array(inputs)
            print(inputs.shape)

            all_data.append(inputs)

        all_data = np.concatenate(all_data, axis=0)
        all_raw_data = np.concatenate(all_raw_data, axis=0)


        return all_data, all_raw_data

class Yolo26(Engine):
    def __str__(self):
        return "YOLO26 ENGINE"

    def get_data(self, user_video=None):
        from ultralytics import YOLO

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
                project="outputs/videos/overlays",
                name="full_overlay.mp4",
                exist_ok=True,
                show=True)

            for curr_frame, result in enumerate(results):
                current_pose = result.keypoints.xy[0].clone()

                # right shoulder angle
                a = (current_pose[8, 0], current_pose[8, 1])
                b = (current_pose[6, 0], current_pose[6, 1])
                c = (current_pose[12, 0], current_pose[12, 1])
                data[curr_frame, 0] =Engine.find_angle(a, b, c)

                # left shoulder angle
                a = (current_pose[7, 0], current_pose[7, 1])
                b = (current_pose[5, 0], current_pose[5, 1])
                c = (current_pose[11, 0], current_pose[11, 1])
                data[curr_frame, 1] =Engine.find_angle(a, b, c)

                # right elbow angle
                a = (current_pose[6, 0], current_pose[6, 1])
                b = (current_pose[8, 0], current_pose[8, 1])
                c = (current_pose[10, 0], current_pose[10, 1])
                data[curr_frame, 2] =Engine.find_angle(a, b, c)

                # left elbow angle
                a = (current_pose[5, 0], current_pose[5, 1])
                b = (current_pose[7, 0], current_pose[7, 1])
                c = (current_pose[9, 0], current_pose[9, 1])
                data[curr_frame, 3] =Engine.find_angle(a, b, c)

                # right hip angle
                a = (current_pose[6, 0], current_pose[6, 1])
                b = (current_pose[12, 0], current_pose[12, 1])
                c = (current_pose[14, 0], current_pose[14, 1])
                data[curr_frame, 4] =Engine.find_angle(a, b, c)

                # left hip angle
                a = (current_pose[5, 0], current_pose[5, 1])
                b = (current_pose[11, 0], current_pose[11, 1])
                c = (current_pose[13, 0], current_pose[13, 1])
                data[curr_frame, 5] =Engine.find_angle(a, b, c)

                # right knee angle
                a = (current_pose[12, 0], current_pose[12, 1])
                b = (current_pose[14, 0], current_pose[14, 1])
                c = (current_pose[16, 0], current_pose[16, 1])
                data[curr_frame, 6] =Engine.find_angle(a, b, c)

                # left knee angle
                a = (current_pose[11, 0], current_pose[11, 1])
                b = (current_pose[13, 0], current_pose[13, 1])
                c = (current_pose[15, 0], current_pose[15, 1])
                data[curr_frame, 7] =Engine.find_angle(a, b, c)

                # # right ankle angle
                # a = (current_pose[26, 0], current_pose[26, 1])
                # b = (current_pose[28, 0], current_pose[28, 1])
                # c = (current_pose[32, 0], current_pose[32, 1])
                # data[curr_frame, 8] =Engine.find_angle(a,b,c)
                #
                # # left ankle angle
                # a = (current_pose[25, 0], current_pose[25, 1])
                # b = (current_pose[27, 0], current_pose[27, 1])
                # c = (current_pose[31, 0], current_pose[31, 1])
                # data[curr_frame, 9] =Engine.find_angle(a,b,c)

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

            all_raw_data.append(data[Engine.border: -Engine.border - 1])

            inputs = []
            for i in range(0, len(data) - Engine.window_size, Engine.step):
                window = data[i:i + Engine.window_size]
                inputs.append(window)

            inputs = np.array(inputs)
            print(inputs.shape)

            all_data.append(inputs)

        all_data = np.concatenate(all_data, axis=0)
        all_raw_data = np.concatenate(all_raw_data, axis=0)

        return all_data, all_raw_data

class MediaPipe(Engine):
    def __str__(self):
        return "MEDIAPIPE ENGINE"

    def get_data(self, user_video=None):
        import mediapipe as mp

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
    
        valid_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]
    
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
    
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            data = np.zeros((frame_count, 50)) # 50 features

            for curr_frame in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
    
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
    
                if results.pose_landmarks:
                    current_pose = results.pose_landmarks.landmark
    
                    # right shoulder angle
                    a = (current_pose[14].x, current_pose[14].y)
                    b = (current_pose[12].x, current_pose[12].y)
                    c = (current_pose[24].x, current_pose[24].y)
                    data[curr_frame, 0] = Engine.find_angle(a,b,c)
    
                    # left shoulder angle
                    a = (current_pose[13].x, current_pose[13].y)
                    b = (current_pose[11].x, current_pose[11].y)
                    c = (current_pose[23].x, current_pose[23].y)
                    data[curr_frame, 1] = Engine.find_angle(a,b,c)
    
                    # right elbow angle
                    a = (current_pose[12].x, current_pose[12].y)
                    b = (current_pose[14].x, current_pose[14].y)
                    c = (current_pose[16].x, current_pose[16].y)
                    data[curr_frame, 2] = Engine.find_angle(a,b,c)
    
                    # left elbow angle
                    a = (current_pose[11].x, current_pose[11].y)
                    b = (current_pose[13].x, current_pose[13].y)
                    c = (current_pose[15].x, current_pose[15].y)
                    data[curr_frame, 3] = Engine.find_angle(a,b,c)
    
                    # right hip angle
                    a = (current_pose[12].x, current_pose[12].y)
                    b = (current_pose[24].x, current_pose[24].y)
                    c = (current_pose[26].x, current_pose[26].y)
                    data[curr_frame, 4] = Engine.find_angle(a,b,c)
    
                    # left hip angle
                    a = (current_pose[11].x, current_pose[11].y)
                    b = (current_pose[23].x, current_pose[23].y)
                    c = (current_pose[25].x, current_pose[25].y)
                    data[curr_frame, 5] = Engine.find_angle(a,b,c)
    
                    # right knee angle
                    a = (current_pose[24].x, current_pose[24].y)
                    b = (current_pose[26].x, current_pose[26].y)
                    c = (current_pose[28].x, current_pose[28].y)
                    data[curr_frame, 6] = Engine.find_angle(a,b,c)
    
                    # left knee angle
                    a = (current_pose[23].x, current_pose[23].y)
                    b = (current_pose[25].x, current_pose[25].y)
                    c = (current_pose[27].x, current_pose[27].y)
                    data[curr_frame, 7] = Engine.find_angle(a,b,c)
    
                    # right ankle angle
                    a = (current_pose[26].x, current_pose[26].y)
                    b = (current_pose[28].x, current_pose[28].y)
                    c = (current_pose[32].x, current_pose[32].y)
                    data[curr_frame, 8] = Engine.find_angle(a,b,c)
    
                    # left ankle angle
                    a = (current_pose[25].x, current_pose[25].y)
                    b = (current_pose[27].x, current_pose[27].y)
                    c = (current_pose[31].x, current_pose[31].y)
                    data[curr_frame, 9] = Engine.find_angle(a,b,c)
    
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

                        raw_lm1 = int(lm1.x * width), int(lm1.y * height)
                        raw_lm2 = int(lm2.x * width), int(lm2.y * height)
                        cv2.line(frame, raw_lm1, raw_lm2, (255,255,255), 4)

                        if user_video:
                            x1 = (lm1.x - center_x) / torso_length
                            y1 = (lm1.y - center_y) / torso_length

                            x1 = int((x1 / 3 + 0.5) * width)
                            y1 = int((y1 / 3 + 0.5) * height)

                            x2 = (lm2.x - center_x) / torso_length
                            y2 = (lm2.y - center_y) / torso_length

                            x2 = int((x2 / 3 + 0.5) * width)
                            y2 = int((y2 / 3 + 0.5) * height)

                            cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

                    if user_video:
                        user_skeleton.write(canvas)
                        user_overlay.write(frame)

                    cv2.imshow('Processing Preview', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    print(f"Frame: {curr_frame + 1}/{frame_count} Saved")

            cap.release()
            if user_video:
                user_skeleton.release()
                user_overlay.release()
                cv2.destroyAllWindows()

            # smooth angle data to reduce noise
            smooth_window = 3
            half_window = smooth_window // 2
            for i in range(half_window, len(data) - half_window):
                data[i, :10] = np.mean(data[i - half_window:i + half_window + 1, :10], axis=0)
    
            # find angular velocities from smoothed angles
            for i in range(1, len(data)):
                data[i, 10:20] = data[i, :10] - data[i-1, :10]
    
            all_raw_data.append(data[Engine.border: -Engine.border-1])

            inputs = []
            for i in range(0, len(data) - Engine.window_size, Engine.step):
                window = data[i:i + Engine.window_size]
                inputs.append(window)
    
            inputs = np.array(inputs)
            print(inputs.shape)
    
            all_data.append(inputs)
    
    
        all_data = np.concatenate(all_data, axis=0)
        all_raw_data = np.concatenate(all_raw_data, axis=0)
    
        return all_data, all_raw_data
    
def get_data(choice, user_video=None):
    if choice == 'yolo26':
        engine = Yolo26()
    elif choice == 'mediapipe':
        engine = MediaPipe()
    else:
        engine = MMPose()

    print(engine, '- INITIALIZED')
    start = time.time()







    data = engine.get_data(user_video)
    print('TIME:', time.time() - start)

    return data