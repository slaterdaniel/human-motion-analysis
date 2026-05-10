import cv2
import numpy as np
import os
from src.pose import Engine


def main():
    video = '../data/user_input/boetest.mov'
    cap = cv2.VideoCapture(video)
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    border = Engine.get_formatting()[1]

    print('FRAMES:', frame_num)
    print('WINDOWS:', frame_num - 1 - border * 2)
    print('BORDER:', border)
    print('Input Phase of Currently Displayed Phase\n\n')

    phase_labels = []
    curr_frame = border
    cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)

    PHASE_STRINGS = ['Right Ground Contact',
                     'Right Propulsion',
                     'Right Flight',
                     'Left Ground Contact',
                     'Left Propulsion',
                     'Left Flight']

    while True:
        ret, frame = cap.read()
        if curr_frame >= frame_num - border - 1:
            break

        cv2.imshow('Labeling Utility', frame)

        key = cv2.waitKey(0) & 0xFF # Wait for keypress per frame

        if curr_frame == border:
            while True:
                print('Starting Phase:')
                if key == ord('0'):
                    label = 0
                    break
                elif key == ord('1'):
                    label = 1
                    break
                elif key == ord('2'):
                    label = 2
                    break
                elif key == ord('3'):
                    label = 3
                    break
                elif key == ord('4'):
                    label = 4
                    break
                elif key == ord('5'):
                    label = 5
                    break

            phase_labels.append(label)
            print(f'Labeled Frame {curr_frame} as "{PHASE_STRINGS[label]}"')
            curr_frame += 1

        elif key == ord('1'):
            label += 1
            if label > 5:
                label = 0
            phase_labels.append(label)
            print(f'Labeled Frame {curr_frame} as "{PHASE_STRINGS[label]}"')
            curr_frame += 1

        elif key == ord('0'):
            phase_labels.append(label)
            print(f'Labeled Frame {curr_frame} as "{PHASE_STRINGS[label]}"')
            curr_frame += 1

        elif key == ord('s'):
            print(f'\nFrame {curr_frame}: {phase_labels}\n')
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)

        else:
            curr_frame -= 1
            print(f'Removed Label "{PHASE_STRINGS[phase_labels.pop()]}" from frame {curr_frame}')
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)

    cv2.destroyAllWindows()
    np.save(f'../assets/video_training_labels/{os.path.splitext(os.path.basename(video))[0]}.npy', phase_labels)
    print(f'\nFinal Labels:\n{np.array(phase_labels)}')
    print(f'Length = {len(phase_labels)}/{frame_num - 1 - border * 2}')


main()