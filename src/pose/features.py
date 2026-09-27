from tkinter.constants import RIGHT

FEATURE_STRINGS = [

    # First 16 Values = Body Angles
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
    'RIGHT TIBIAL ANGLE',
    'LEFT TIBIAL ANGLE',
    'RIGHT FOOT INCLINATION ANGLE',
    'LEFT FOOT INCLINATION ANGLE',
    'RIGHT FORWARD LEAN ANGLE',
    'LEFT FORWARD LEAN ANGLE',

    # Next 16 Values = Body Angle Velocities
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
    'RIGHT TIBIAL ANGLE VELOCITY',
    'LEFT TIBIAL ANGLE VELOCITY',
    'RIGHT FOOT INCLINATION ANGLE VELOCITY',
    'LEFT FOOT INCLINATION ANGLE VELOCITY',
    'RIGHT FORWARD LEAN ANGLE VELO',
    'LEFT FORWARD LEAN ANGLE VELO',

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
    0:  (4, 5),   # Right Shoulder  (angle)
    1:  (2, 3),   # Left Shoulder   (angle)
    2:  (8, 9),   # Right Elbow     (angle)
    3:  (6, 7),   # Left Elbow      (angle)
    4:  (16, 17), # Right Hip       (angle)
    5:  (14, 15), # Left Hip        (angle)
    6:  (20, 21), # Right Knee      (angle)
    7:  (18, 19), # Left Knee       (angle)
    8:  (24, 25), # Right Ankle     (angle)
    9:  (22, 23), # Left Ankle      (angle)
    10: (24, 25), # Right Tibia     (angle)
    11: (22, 23), # Left Tibia      (angle)
    12: (28, 29), # Right Foot Incl (angle)
    13: (26, 27), # Left Foot Incl  (angle)
    14: (4, 5),   # Right F-Lean    (angle)
    15: (2, 3),   # Left F-Lean     (angle)
    16: (4, 5),   # Right Shoulder  (angle velo)
    17: (2, 3),   # Left Shoulder   (angle velo)
    18: (8, 9),   # Right Elbow     (angle velo)
    19: (6, 7),   # Left Elbow      (angle velo)
    20: (16, 17), # Right Hip       (angle velo)
    21: (14, 15), # Left Hip        (angle velo)
    22: (20, 21), # Right Knee      (angle velo)
    23: (18, 19), # Left Knee       (angle velo)
    24: (24, 25), # Right Ankle     (angle velo)
    25: (22, 23), # Left Ankle      (angle velo)
    26: (24, 25), # Right Tibia     (angle velo)
    27: (22, 23), # Left Tibia      (angle velo)
    28: (28, 29), # Right Foot Incl (angle velo)
    29: (26, 27), # Left Foot Incl  (angle velo)
    30: (4, 5),   # Right F-Lean    (angle velo)
    31: (2, 3),   # Left F-Lean     (angle velo)
    32: (0, 1),   # Nose
    33: (0, 1),   # Nose
    34: (2, 3),   # Left Shoulder
    35: (2, 3),   # Left Shoulder
    36: (4, 5),   # Right Shoulder
    37: (4, 5),   # Right Shoulder
    38: (6, 7),   # Left Elbow
    39: (6, 7),   # Left Elbow
    40: (8, 9),   # Right Elbow
    41: (8, 9),   # Right Elbow
    42: (10, 11), # Left Wrist
    43: (10, 11), # Left Wrist
    44: (12, 13), # Right Wrist
    45: (12, 13), # Right Wrist
    46: (14, 15), # Left Hip
    47: (14, 15), # Left Hip
    48: (16, 17), # Right Hip
    49: (16, 17), # Right Hip
    50: (18, 19), # Left Knee
    51: (18, 19), # Left Knee
    52: (20, 21), # Right Knee
    53: (20, 21), # Right Knee
    54: (22, 23), # Left Ankle
    55: (22, 23), # Left Ankle
    56: (24, 25), # Right Ankle
    57: (24, 25), # Right Ankle
    58: (26, 27), # Left Foot
    59: (26, 27), # Left Foot
    60: (28, 29), # Right Foot
    61: (28, 29), # Right Foot
}

running_phase_strings = [
        "rgc",
        "rp",
        "rf",
        "lgc",
        "lp",
        "lf"
    ]

R_SHOULDER_ANGLE = 0
L_SHOULDER_ANGLE = 1
R_ELBOW_ANGLE    = 2
L_ELBOW_ANGLE    = 3
R_HIP_ANGLE      = 4
L_HIP_ANGLE      = 5
R_KNEE_ANGLE     = 6
L_KNEE_ANGLE     = 7
R_ANKLE_ANGLE    = 8
L_ANKLE_ANGLE    = 9
R_TIBIAL_ANGLE   = 10
L_TIBIAL_ANGLE   = 11
R_FOOT_INCLINATION_ANGLE = 12
L_FOOT_INCLINATION_ANGLE = 13
R_FORWARD_LEAN = 14
L_FORWARD_LEAN = 15

R_SHOULDER_ANGLE_VEL = 16
L_SHOULDER_ANGLE_VEL = 17
R_ELBOW_ANGLE_VEL    = 18
L_ELBOW_ANGLE_VEL    = 19
R_HIP_ANGLE_VEL      = 20
L_HIP_ANGLE_VEL      = 21
R_KNEE_ANGLE_VEL     = 22
L_KNEE_ANGLE_VEL     = 23
R_ANKLE_ANGLE_VEL    = 24
L_ANKLE_ANGLE_VEL    = 25
R_TIBIAL_ANGLE_VEL   = 26
L_TIBIAL_ANGLE_VEL   = 27
R_FOOT_INCLINATION_ANGLE_VEL = 28
L_FOOT_INCLINATION_ANGLE_VEL = 29
R_FORWARD_LEAN_VEL = 30
L_FORWARD_LEAN_VEL = 31

NOSE_X       = 32
NOSE_Y       = 33
L_SHOULDER_X = 34
L_SHOULDER_Y = 35
R_SHOULDER_X = 36
R_SHOULDER_Y = 37
L_ELBOW_X    = 38
L_ELBOW_Y    = 39
R_ELBOW_X    = 40
R_ELBOW_Y    = 41
L_WRIST_X    = 42
L_WRIST_Y    = 43
R_WRIST_X    = 44
R_WRIST_Y    = 45
L_HIP_X      = 46
L_HIP_Y      = 47
R_HIP_X      = 48
R_HIP_Y      = 49
L_KNEE_X     = 50
L_KNEE_Y     = 51
R_KNEE_X     = 52
R_KNEE_Y     = 53
L_ANKLE_X    = 54
L_ANKLE_Y    = 55
R_ANKLE_X    = 56
R_ANKLE_Y    = 57
L_FOOT_X     = 58
L_FOOT_Y     = 59
R_FOOT_X     = 60
R_FOOT_Y     = 61