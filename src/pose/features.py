FEATURE_STRINGS = [

    # First 10 Values = Body Angles
    'RIGHT SHOULDER ANGLE',
    'LEFT SHOULDER ANGLE',
    'RIGHT ELBOW ANGLE',
    'LEFT ELBOW ANGLE',
    'RIGHT HIP ANGLE',
    'LEFT HIP ANGLE',
    'RIGHT KNEE ANGLE',
    'LEFT KNEE ANGLE',
    'RIGHT ANKLE ANGLE',
    'LEFT ANKLE ANGLE',

    # Next 10 Values = Body Angle Velocities
    'RIGHT SHOULDER ANGLE VELOCITY',
    'LEFT SHOULDER ANGLE VELOCITY',
    'RIGHT ELBOW ANGLE VELOCITY',
    'LEFT ELBOW ANGLE VELOCITY',
    'RIGHT HIP ANGLE VELOCITY',
    'LEFT HIP ANGLE VELOCITY',
    'RIGHT KNEE ANGLE VELOCITY',
    'LEFT KNEE ANGLE VELOCITY',
    'RIGHT ANKLE ANGLE VELOCITY',
    'LEFT ANKLE ANGLE VELOCITY',

    # Final 30 = mediapipe landmark coordinates
    'NOSE X',
    'NOSE Y',
    'LEFT SHOULDER X',
    'LEFT SHOULDER Y',
    'RIGHT SHOULDER X',
    'RIGHT SHOULDER Y',
    'LEFT ELBOW X',
    'LEFT ELBOW Y',
    'RIGHT ELBOW X',
    'RIGHT ELBOW Y',
    'LEFT WRIST X',
    'LEFT WRIST Y',
    'RIGHT WRIST X',
    'RIGHT WRIST Y',
    'LEFT HIP X',
    'LEFT HIP Y',
    'RIGHT HIP X',
    'RIGHT HIP Y',
    'LEFT KNEE X',
    'LEFT KNEE Y',
    'RIGHT KNEE X',
    'RIGHT KNEE Y',
    'LEFT ANKLE X',
    'LEFT ANKLE Y',
    'RIGHT ANKLE X',
    'RIGHT ANKLE Y',
    'LEFT FOOT X',
    'LEFT FOOT Y',
    'RIGHT FOOT X',
    'RIGHT FOOT Y',
]

COORDINATE_PAIRS = {
    0:  (4, 5),   # Right Shoulder (angle)
    1:  (2, 3),   # Left Shoulder  (angle)
    2:  (8, 9),   # Right Elbow    (angle)
    3:  (6, 7),   # Left Elbow     (angle)
    4:  (16, 17), # Right Hip      (angle)
    5:  (14, 15), # Left Hip       (angle)
    6:  (20, 21), # Right Knee     (angle)
    7:  (18, 19), # Left Knee      (angle)
    8:  (24, 25), # Right Ankle    (angle)
    9:  (22, 23), # Left Ankle     (angle)
    10: (4, 5),   # Right Shoulder (angle velo)
    11: (2, 3),   # Left Shoulder  (angle velo)
    12: (8, 9),   # Right Elbow    (angle velo)
    13: (6, 7),   # Left Elbow     (angle velo)
    14: (16, 17), # Right Hip      (angle velo)
    15: (14, 15), # Left Hip       (angle velo)
    16: (20, 21), # Right Knee     (angle velo)
    17: (18, 19), # Left Knee      (angle velo)
    18: (24, 25), # Right Ankle    (angle velo)
    19: (22, 23), # Left Ankle     (angle velo)
    20: (0, 1),   # Nose
    21: (0, 1),   # Nose
    22: (2, 3),   # Left Shoulder
    23: (2, 3),   # Left Shoulder
    24: (4, 5),   # Right Shoulder
    25: (4, 5),   # Right Shoulder
    26: (6, 7),   # Left Elbow
    27: (6, 7),   # Left Elbow
    28: (8, 9),   # Right Elbow
    29: (8, 9),   # Right Elbow
    30: (10, 11), # Left Wrist
    31: (10, 11), # Left Wrist
    32: (12, 13), # Right Wrist
    33: (12, 13), # Right Wrist
    34: (14, 15), # Left Hip
    35: (14, 15), # Left Hip
    36: (16, 17), # Right Hip
    37: (16, 17), # Right Hip
    38: (18, 19), # Left Knee
    39: (18, 19), # Left Knee
    40: (20, 21), # Right Knee
    41: (20, 21), # Right Knee
    42: (22, 23), # Left Ankle
    43: (22, 23), # Left Ankle
    44: (24, 25), # Right Ankle
    45: (24, 25), # Right Ankle
    46: (26, 27), # Left Foot
    47: (26, 27), # Left Foot
    48: (28, 29), # Right Foot
    49: (28, 29), # Right Foot
}

running_phase_strings = [
        "rgc",
        "rp",
        "rf",
        "lgc",
        "lp",
        "lf"
    ]