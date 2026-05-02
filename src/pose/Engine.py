import cv2
import numpy as np
import os
import ffmpeg

def get_formatting():
    window_size = 9
    border = window_size // 2
    step = 1
    return window_size, border, step

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

    return videos

def apply_filters(video):
    (
        ffmpeg
        .input(video)
        # 3. Boost contrast to help the AI see limbs against the treadmill
        .filter('eq', contrast=1.3, brightness=0.02)
        .output('../assets/current_video.mp4', pix_fmt='yuv420p', crf=18)
        .run(overwrite_output=True)
    )

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