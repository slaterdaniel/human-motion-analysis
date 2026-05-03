import cv2
import numpy as np
import os
from src.pose import Engine

video = '../data/user_input/boetest.MOV'
cap = cv2.VideoCapture(video)
frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
border = Engine.get_formatting()[1]

print('FRAMES:', frame_num)
print('BORDER:', border)
print('Input Phase of Currently Displayed Phase:\n\n')
phase_labels = []
curr_frame = border
cap.set(cv2.CAP_PROP_POS_FRAMES, border)

PHASE_STRINGS = {0: 'Right Ground Contact',
                 1: 'Right Propulsion',
                 2: 'Right Flight',
                 3: 'Left Ground Contact',
                 4: 'Left Propulsion',
                 5: 'Left Flight'}

while True:
    ret, frame = cap.read()
    if not ret or curr_frame >= frame_num - border - 2:
        break

    cv2.imshow('Labeling Utility', frame)

    key = cv2.waitKey(0) & 0xFF # Wait for keypress per frame

    if key == ord('0'):
        label = 0
    elif key == ord('1'):
        label = 1
    elif key == ord('2'):
        label = 2
    elif key == ord('3'):
        label = 3
    elif key == ord('4'):
        label = 4
    elif key == ord('5'):
        label = 5
    elif key == ord('s'):
        print(f'\nFrame {curr_frame}: {phase_labels}\n')
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        continue
    else:
        curr_frame -= 1
        print(f'Removed Label "{PHASE_STRINGS[phase_labels.pop()]}" from frame {curr_frame}')
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        continue

    phase_labels.append(label)
    print(f'Labeled frame {curr_frame} as "{PHASE_STRINGS[label]}"')

    curr_frame += 1

cv2.destroyAllWindows()
np.save(f'../assets/labels/{os.path.splitext(os.path.basename(video))[0]}.npy', phase_labels)
print(f'\nFinal Labels:\n{np.array(phase_labels)}')
