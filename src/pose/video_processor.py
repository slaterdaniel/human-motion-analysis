import cv2
import numpy as np
import os
import ffmpeg
import sys
from unittest.mock import MagicMock
import time

mock_ext = MagicMock()
mock_ext.__spec__ = MagicMock()
sys.modules['mmcv._ext'] = mock_ext

from mmpose.apis import inference_topdown, init_model

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

    config_file = "../mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py"
    checkpoint_file = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"

    model = init_model(config_file, checkpoint_file, device='cpu')
    connections = model.dataset_meta['skeleton_links'][:25]
    connection_colors = model.dataset_meta['skeleton_link_colors']

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
        (
            ffmpeg
            .input(video)
            # 3. Boost contrast to help the AI see limbs against the treadmill
            .filter('eq', contrast=1.3, brightness=0.02)
            .output('../assets/current_video.mp4', pix_fmt='yuv420p', crf=18)
            .run(overwrite_output=True)
        )
        cap = cv2.VideoCapture('../assets/current_video.mp4')
        # cap = cv2.VideoCapture(video)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if user_video:
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

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        data = np.zeros((frame_count, 50)) # 50 features
        start = time.time()

        for curr_frame in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = inference_topdown(model, frame)  # Run the model
            
            # Extract data
            keypoints = results[0].pred_instances.keypoints[0].astype(np.float64)  # Array of [x, y]

            # for i, lm in enumerate(current_pose):
            #     if prev_pose:
            #         alpha = 0.5
            #         prev_lm = prev_pose[i]
            #         lm.x = alpha * prev_lm.x + (1 - alpha) * lm.x
            #         lm.y = alpha * prev_lm.y + (1 - alpha) * lm.y

            # right shoulder angle
            a = keypoints[8]
            b = keypoints[6]
            c = keypoints[12]
            data[curr_frame, 0] = find_angle(a,b,c)

            # left shoulder angle
            a = keypoints[7]
            b = keypoints[5]
            c = keypoints[11]
            data[curr_frame, 1] = find_angle(a,b,c)

            # right elbow angle
            a = keypoints[6]
            b = keypoints[8]
            c = keypoints[10]
            data[curr_frame, 2] = find_angle(a,b,c)

            # left elbow angle
            a = keypoints[5]
            b = keypoints[7]
            c = keypoints[9]
            data[curr_frame, 3] = find_angle(a,b,c)

            # right hip angle
            a = keypoints[6]
            b = keypoints[12]
            c = keypoints[14]
            data[curr_frame, 4] = find_angle(a,b,c)

            # left hip angle
            a = keypoints[5]
            b = keypoints[11]
            c = keypoints[13]
            data[curr_frame, 5] = find_angle(a,b,c)

            # right knee angle
            a = keypoints[12]
            b = keypoints[14]
            c = keypoints[16]
            data[curr_frame, 6] = find_angle(a,b,c)

            # left knee angle
            a = keypoints[11]
            b = keypoints[13]
            c = keypoints[15]
            data[curr_frame, 7] = find_angle(a,b,c)

            # right ankle angle
            a = keypoints[14]
            b = keypoints[16]
            c = keypoints[21]
            data[curr_frame, 8] = find_angle(a,b,c)

            # left ankle angle
            a = keypoints[13]
            b = keypoints[15]
            c = keypoints[18]
            data[curr_frame, 9] = find_angle(a,b,c)

            # center coordinates around waist to normalize data across user_input
            center_x = (keypoints[11, 0] + keypoints[12, 0]) / 2
            center_y = (keypoints[11, 1] + keypoints[12, 1]) / 2

            # scale by torso length so different size people can be compared
            right_torso = np.linalg.norm(keypoints[6] - keypoints[12])
            left_torso = np.linalg.norm(keypoints[5] - keypoints[11])

            torso_length = (right_torso + left_torso) / 2
            zoom = height * .25
            for i, lm in enumerate(np.vstack((keypoints[0], keypoints[5:17], keypoints[18], keypoints[21]))):
                x = (lm[0] - center_x) / torso_length
                y = (lm[1] - center_y) / torso_length

                data[curr_frame, 20 + i*2] = x
                data[curr_frame, 21 + i*2] = y

            if user_video:
                # Initialize blank canvas
                canvas = np.zeros((height, width, 3), dtype=np.uint8)

                # Draw circles for hands
                rhand_x = int(((keypoints[9, 0] - center_x) * zoom / torso_length) + (width / 2))
                rhand_y = int(((keypoints[9, 1] - center_y) * zoom / torso_length) + (height / 2))
                cv2.circle(canvas, (rhand_x, rhand_y), 10, tuple(map(int, connection_colors[10])), -1)

                lhand_x = int(((keypoints[10, 0] - center_x) * zoom / torso_length) + (width / 2))
                lhand_y = int(((keypoints[10, 1] - center_y) * zoom / torso_length) + (height / 2))
                cv2.circle(canvas, (lhand_x, lhand_y), 10, tuple(map(int, connection_colors[11])), -1)

                # Draw skeleton connection
                for i, (pt1, pt2) in enumerate(connections):
                    lm1, lm2 = keypoints[pt1], keypoints[pt2]

                    x1 = int(((lm1[0] - center_x) * zoom / torso_length) + (width / 2))
                    y1 = int(((lm1[1] - center_y) * zoom / torso_length) + (height / 2))

                    x2 = int(((lm2[0] - center_x) * zoom / torso_length) + (width / 2))
                    y2 = int(((lm2[1] - center_y) * zoom / torso_length) + (height / 2))

                    color = tuple(map(int, connection_colors[i]))
                    cv2.line(canvas, (x1, y1), (x2, y2), color, 2)
                user_skeleton.write(canvas)

            for i, (pt1, pt2) in enumerate(connections):
                lm1, lm2 = keypoints[pt1].astype(int), keypoints[pt2].astype(int)
                color = tuple(map(int, connection_colors[i]))
                cv2.line(frame, lm1, lm2, color, 2)

            cv2.imshow('Processing Preview', frame)
            user_overlay.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print(f"Frame: {curr_frame+1}/{frame_count} Saved")

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

        all_raw_data.append(data[border: -border-1])

        inputs = []
        for i in range(0, len(data) - window_size, step):
            window = data[i:i + window_size]
            inputs.append(window)

        inputs = np.array(inputs)
        print(inputs.shape)

        all_data.append(inputs)
        print('TIME:', time.time() - start)

    all_data = np.concatenate(all_data, axis=0)
    all_raw_data = np.concatenate(all_raw_data, axis=0)
    return all_data, all_raw_data