from tensorflow.keras.models import load_model
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go
import os
from pose import Engine

# KEY:
# rgc = right ground contact
# rp  = right propulsion phase
# rf  = right flight phase
# lgc = left ground contact
# lp  = left propulsion phase
# lf  = left flight phase

def find_phase_scores(frame_scores, user_predictions, window_size=9):
    """
    Finds phase scores of the users data
    Args:
        frame_scores: numpy array of features z scores where: rows = features, columns = frame
        user_predictions: numpy array of the model's predictions of which phase the user is in for each frame
        window_size: window size of 1d cnn neural network
    Returns:
        numpy array of the average sum of the squared z scores
        of all features across frames in an instance of a phase
    """
    # Find phase lengths
    current_length = 0
    last_phase = user_predictions[0]
    phase_lengths = []

    for current_phase in user_predictions:
        if current_phase == last_phase:
            current_length += 1
        else:
            phase_lengths.append(current_length)
            current_length = 1
            last_phase = current_phase
    phase_lengths.append(current_length)

    # find scores of each phase
    progress = 0
    total_scores = []
    for length in phase_lengths:
        score = np.sum(frame_scores[:, progress:progress + length] ** 2)
        total_scores.append((score / length, progress + (window_size//2)))
        progress += length

    return np.array(total_scores), np.array(phase_lengths)

def save_video(file, worst_frame, worst_length, best_frame, best_length, raw_data, user_predictions, scored_data, FEATURE_STRINGS, window_size=9):
    """
    Saves video of the user's best instance of a phase overlaid on the worst instance
    Args:
        file: file name to save the video
        worst_frame: starting frame of the worst instance of the phase
        worst_length: length of the worst instance of the phase
        best_frame: starting frame of the worst instance of the phase
        best_length: length of the worst instance of the phase
        raw_data: numpy array of raw values from features in user video
        user_predictions: numpy array of the model's predictions of which phase the user is in for each frame
        scored_data: numpy array of the squared z scores of each feature across each frame
        window_size: window size of 1d cnn neural network
    Returns:
        None
    """
    skeleton_cap = cv2.VideoCapture('../outputs/videos/user_skeleton/user_skeleton.mp4')
    overlay_cap = cv2.VideoCapture('../outputs/videos/overlays/full_overlay.mp4')
    fps = 1.5
    width = int(skeleton_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(skeleton_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    font = cv2.FONT_HERSHEY_SIMPLEX
    style = cv2.LINE_AA

    vertical = True if height > width else False

    PHASE_STRINGS = ['Right Ground Contact',
                     'Right Propulsion',
                     'Right Flight',
                     'Left Ground Contact',
                     'Left Propulsion',
                     'Left Flight']

    border = window_size // 2
    length = np.maximum(worst_length, best_length)

    offset = length // 2 + 1
    worst_start_frame = worst_frame - offset
    best_start_frame = best_frame - offset

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    if vertical:
        feedback_column_dims = height, 250
        out_dims = width * 2 + 250, height
    else:
        feedback_column_dims = 250, width
        out_dims = width, height*2 + 250

    # feedback_column_dims = 250, height if vertical else width, 250
    # out_dims = (width*2 + feedback_column_dims[0], height) if vertical else (width + feedback_column_dims[1], height*2)
    out = cv2.VideoWriter(file, fourcc, fps, out_dims)
    feedback_box = np.zeros((*feedback_column_dims, 3), dtype=np.uint8)
    worst = []
    user_overlay = []

    overlay_cap.set(cv2.CAP_PROP_POS_FRAMES, worst_start_frame)
    skeleton_cap.set(cv2.CAP_PROP_POS_FRAMES, worst_start_frame)
    for i in range(length + (offset * 2)):
        ret1, overlay_frame = overlay_cap.read()
        ret2, skeleton_frame = skeleton_cap.read()
        if not (ret1 and ret2) or worst_start_frame + i - border >= scored_data.shape[1]:
            break

        if offset <= i < offset + worst_length:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        curr_frame = int(worst_start_frame + i)
        skeleton_frame = np.clip(skeleton_frame * [0.5, 0.5, 1], 0, 255).astype(np.uint8)

        frame_scores = scored_data[:, curr_frame - border]
        score = round(np.sum(frame_scores ** 2), 1)
        biggest_mistake = np.argmax(np.abs(frame_scores))

        error_value = round(raw_data[biggest_mistake, curr_frame - border], 2)
        reference_value = round(raw_data[biggest_mistake, int(best_start_frame + i) - border], 2)

        skeleton_frame = cv2.hconcat([skeleton_frame, feedback_box]) if vertical else cv2.vconcat([skeleton_frame, feedback_box])

        if vertical:
            cv2.putText(skeleton_frame,
                        f"Phase: {PHASE_STRINGS[user_predictions[curr_frame - border]]}",
                        (int(width * .8), int(height * .45)), font, 1, color, 2, style)

            cv2.putText(skeleton_frame,
                        f"Score: {score}",
                        (int(width * .8), int(height * .5)), font, 1, color, 2, style)

            cv2.putText(skeleton_frame,
                        f"{FEATURE_STRINGS[biggest_mistake]}:",
                        (int(width * .8), int(height * .6)), font, 1, (255, 255, 255), 2, style)

            cv2.putText(skeleton_frame,
                        f"{'Error:':<8} {error_value:<8} Frame: {curr_frame + 1}",
                        (int(width * .8), int(height * .65)), font, 1, (127,127,255), 2, style)

            cv2.putText(skeleton_frame,
                        f"{'Ref:':<8} {reference_value:<8} Frame: {int(best_start_frame + i + 1)}",
                        (int(width * .8), int(height * .7)), font, 1, (127, 255, 127), 2, style)

            cv2.putText(skeleton_frame,
                        f"{'% Error:':<8} {round((error_value - reference_value) * 100 / (reference_value + 1e-8), 2)}%",
                        (int(width * .8), int(height * .75)), font, 1, (127, 255, 255), 2, style)
        else:
            cv2.putText(skeleton_frame,
                        f"Phase: {PHASE_STRINGS[user_predictions[curr_frame - border]]}",
                        (int(width * .25), int(height * .975)), font, 1, color, 2, style)

            cv2.putText(skeleton_frame,
                        f"Score: {score}",
                        (int(width * .25), int(height * 1.05)), font, 1, color, 2, style)

            cv2.putText(skeleton_frame,
                        f"{FEATURE_STRINGS[biggest_mistake]}:",
                        (int(width * .55), int(height * .925)), font, 1, (255, 255, 255), 2, style)

            cv2.putText(skeleton_frame,
                        f"{'Error:':<8} {error_value:<8} Frame: {curr_frame + 1}",
                        (int(width * .55), int(height * .975)), font, 1, (127, 127, 255), 2, style)

            cv2.putText(skeleton_frame,
                        f"{'Ref:':<8} {reference_value:<8} Frame: {int(best_start_frame + i + 1)}",
                        (int(width * .55), int(height*1.05)), font, 1, (127, 255, 127), 2, style)

            cv2.putText(skeleton_frame,
                        f"{'% Error:':<8} {round((error_value - reference_value) * 100 / (reference_value + 1e-8), 2)}%",
                        (int(width * .55), int(height * 1.125)), font, 1, (127, 255, 255), 2, style)

        user_overlay.append(overlay_frame)
        worst.append(skeleton_frame)

    skeleton_cap.set(cv2.CAP_PROP_POS_FRAMES, best_start_frame)
    for i in range(len(worst)):
        ret, frame = skeleton_cap.read()
        if not ret:
            break
        frame = np.clip(frame * [0.5, 1, 0.5], 0, 255).astype(np.uint8)
        frame = cv2.hconcat([frame, feedback_box]) if vertical else cv2.vconcat([frame, feedback_box])
        skeleton_overlay = cv2.addWeighted(worst[i], 1, frame, 1, 0)

        concatenate = cv2.hconcat if vertical else cv2.vconcat
        final_frame = concatenate([user_overlay[i], skeleton_overlay])
        out.write(final_frame)

    overlay_cap.release()
    skeleton_cap.release()
    out.release()
    print(f"{os.path.basename(file):<25} Successfully Saved")

def main():
    USER_VIDEO = "../data/user_input/jose-test.mov" # ***REPLACE WITH FILE OF USER VIDEO***

    while True:
        engine = input("Engine to Use: ").lower()
        if engine == 'mediapipe':
            from pose import mediapipe_video_processor as video_processor
            phase_classifier = '../assets/phase_classifier_models/mediapipe_phase_classifier.keras'
            break
        elif engine == 'yolo26':
            from pose import yolo26_video_processor as video_processor
            phase_classifier = '../assets/phase_classifier_models/yolo26_phase_classifier.keras'
            break
        elif engine == 'mmpose':
            from pose import mmpose_video_processor as video_processor
            phase_classifier = '../assets/phase_classifier_models/mmpose_phase_classifier.keras'
            break
        else:
            print("Invalid Engine Inputted. Try Again.\n")

    graph_saving = True if input('Save Graphs?\n[y/n]\n')[0] == 'y' else False

    np.set_printoptions(threshold=np.inf, suppress=True, precision=3, linewidth=95)

    # user data = array formatted for 1D CNN 9 frame windows
    # raw user data = array where shape=[feature, frame]
    user_data, raw_data = video_processor.get_data(show=False, user_video=USER_VIDEO)
    raw_data = raw_data.T

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

    # Median Absolute Deviation (MAD) and medians of features in reference data
    with open('../assets/phase_statistics/phase_statistics.pkl', 'rb') as f:
        phase_stats = pickle.load(f)

    model = load_model(phase_classifier, compile=False) # 1D CNN to predict phases
    user_predictions = np.argmax(model.predict(user_data), axis=1) # predict the phases from users data
    print(f'\nPhase Predictions:\n{user_predictions}\n')

    # Ground Contact Data
    rgc_starts = []
    lgc_starts = []
    right_contact_lengths = []
    left_contact_lengths = []
    for i in range(1, len(user_predictions)):
        if user_predictions[i] == 0 and  user_predictions[i-1] != 0:
            rgc_starts.append(i)
        elif user_predictions[i] == 3 and  user_predictions[i-1] != 3:
            lgc_starts.append(i)

        elif user_predictions[i] == 2 and user_predictions[i-1] == 1 and len(rgc_starts):
            right_contact_lengths.append(i - rgc_starts[-1])
        elif user_predictions[i] == 5 and user_predictions[i-1] == 4 and len(lgc_starts):
            left_contact_lengths.append(i - lgc_starts[-1])

    # Number of frames right/left foot is on ground each rep
    right_contact_lengths = np.array(right_contact_lengths)
    left_contact_lengths = np.array(left_contact_lengths)

    # X values of right/left foot on initial ground contact
    rfoot = 40 if engine == 'yolo26' else 48
    lfoot = 38 if engine == 'yolo26' else 46
    right_contacts = raw_data[rfoot, rgc_starts]
    left_contacts = raw_data[lfoot, lgc_starts]

    # Number of frames with right/left foot grounded
    right_on_ground = len(user_predictions[user_predictions == 0]) + len(user_predictions[user_predictions == 1])
    left_on_ground = len(user_predictions[user_predictions == 3]) + len(user_predictions[user_predictions == 4])

    scored_data = np.zeros(raw_data.shape) # initialize empty array for storing scored user data

    phase_strings = [
        "rgc",
        "rp",
        "rf",
        "lgc",
        "lp",
        "lf"
    ]
    last = user_predictions[0]
    length = 0 # phase length

    # Score user data using MAD Z-score
    # Each phase is split into "early", "middle", and "late" subphases for comparison to reference user_input
    for i, current in enumerate(user_predictions):
        if current == last and not i == len(user_predictions)-1:
            length += 1
        else:
            if i == len(user_predictions)-1:
                length += 1
                i += 1

            phase = phase_strings[last]

            start = i - length # start frame of the current phase being scored

            base = length // 3 # base size for each subphase
            remainder = length % 3 # extra frames of each phase are given to the "middle" phase

            # save MAD Z-score of subphases
            # "early" subphase
            scored_data[:, start:start + base] = (
                    0.6745 * (raw_data[:, start:start + base] - phase_stats[engine][phase]["early"]["median"])
                    / phase_stats[engine][phase]["early"]["mad"])

            # "middle" subphase
            scored_data[:, start + base:start + base * 2 + remainder] = (
                    0.6745 * (raw_data[:, start + base : start + base * 2 + remainder] - phase_stats[engine][phase]["middle"]["median"])
                    / phase_stats[engine][phase]["middle"]["mad"])

            # "late" subphase
            scored_data[:, start + base * 2 + remainder:i] = (
                    0.6745 * (raw_data[:, start + base * 2 + remainder : start + length] - phase_stats[engine][phase]["late"]["median"])
                    / phase_stats[engine][phase]["late"]["mad"])

            length = 1
        last = current

    # ==============================================================================
    #                            Total Phase Analysis
    # ==============================================================================

    # Score each feature across each frame using MAD Z score
    phase_scores, phase_lengths = find_phase_scores(scored_data, user_predictions)
    np.save('../assets/test_data.npy', phase_scores)

    # Create array of the order each phase appears throughout the user's video
    phase_order = [user_predictions[0]]
    for p in user_predictions[1:]:
        if p != phase_order[-1]:
            phase_order.append(p)
    phase_order = np.array(phase_order)

    # remove first and last phases to ensure they are not cut off by the video
    phase_order[0] = 6 # set to 6 because phases are only 0-5
    phase_order[-1] = 6

    # find the indexes of each phase
    rgc_phase_index = phase_order == 0
    rp_phase_index  = phase_order == 1
    rf_phase_index  = phase_order == 2
    lgc_phase_index = phase_order == 3
    lp_phase_index  = phase_order == 4
    lf_phase_index  = phase_order == 5

    # Find how user form breaks down by taking the median of
    # the 10 "worst" scored instances of each phase on a radar graph
    rgc_breakdown = phase_scores[rgc_phase_index, 0]
    rgc_breakdown.sort()
    rgc_breakdown = np.mean(rgc_breakdown[-3:]) / rgc_breakdown[0]

    rp_breakdown = phase_scores[rp_phase_index, 0]
    rp_breakdown.sort()
    rp_breakdown = np.mean(rp_breakdown[-3:]) / rp_breakdown[0]

    rf_breakdown = phase_scores[rf_phase_index, 0]
    rf_breakdown.sort()
    rf_breakdown = np.mean(rf_breakdown[-3:]) / rf_breakdown[0]

    lgc_breakdown = phase_scores[lgc_phase_index, 0]
    lgc_breakdown.sort()
    lgc_breakdown = np.mean(lgc_breakdown[-3:]) / lgc_breakdown[0]

    lp_breakdown = phase_scores[lp_phase_index, 0]
    lp_breakdown.sort()
    lp_breakdown = np.mean(lp_breakdown[-3:]) / lp_breakdown[0]

    lf_breakdown = phase_scores[lf_phase_index, 0]
    lf_breakdown.sort()
    lf_breakdown = np.mean(lf_breakdown[-3:]) / lf_breakdown[0]

    print(f"{'Phase Breakdown Data':<25} Successfully Saved\n")

    # Phase Z-Scores over time
    plt.figure(figsize=(18, 5))
    plt.plot(phase_scores[rgc_phase_index, 0], label='Right Ground Contact')
    plt.plot(phase_scores[rp_phase_index, 0], label='Right Propulsion')
    plt.plot(phase_scores[rf_phase_index, 0], label='Right Flight')
    plt.plot(phase_scores[lgc_phase_index, 0], label='Left Ground Contact')
    plt.plot(phase_scores[lp_phase_index, 0], label='Left Propulsion')
    plt.plot(phase_scores[lf_phase_index, 0], label='Left Flight')
    plt.xlabel('Rep #')
    plt.ylabel('Phase Z-Score')
    plt.legend()
    plt.savefig('../outputs/graphs/phase_breakdown/Phase_Z-Scores.png', dpi=300)

    # Create array of the indexes of each rep as they appear in the user's video
    rep_index = []
    curr = 1
    for i in range(len(phase_order)):
        if phase_order[i] == 0 or phase_order[i] == 3:
            curr += 1
        rep_index.append(curr)
    rep_index = np.array(rep_index)

    # Save stride Frequency Data
    steps_per_minute = (60 * rep_index[-1]) / (len(user_predictions) / 30)
    strides_per_sec = []
    window = 30

    # Slide 1-second window to track strides per second over time
    for i in range(len(user_predictions)):
        start = np.maximum(0, i - window // 2)
        end = np.minimum(len(user_predictions), i + window // 2)
        last_phase = user_predictions[start]
        phase_count = 0

        # # count number of frames within window
        # for phase in user_predictions[start+1:end]:
        #     if (phase == 0 or phase == 3) and phase != last_phase: # Count rep when left or right foot hits the ground
        #         count += 1
        #     last_phase = phase
        # strides_per_sec.append(count * window / (end-start)) # Adjust count to window size

        for phase in user_predictions[start+1:end]:
            if phase != last_phase:
                phase_count += 1
            last_phase = phase
        strides_per_sec.append((phase_count / 3) * window / (end-start))

    # phases are deemed "faulty" when the mean of the sum of the features squared z scores
    # in a given phase are in higher than the 95th percentile of phase z scores
    cutoff = np.percentile(phase_scores[:,0], 95)
    faulty_phases = phase_scores[:, 0] > cutoff

    # Create a dictionary of the starting frames of each "faulty" phase
    faulty_phase_frames = {key: [] for key in range(6)}
    for i in range(len(faulty_phases)):
        phase = phase_order[i]
        if faulty_phases[i] and phase in range(6):
            faulty_phase_frames[phase].append(int(phase_scores[i, 1]))

    # ==============================================================================
    #                                 OUTPUT DATA
    # ==============================================================================

    # Save Phase Analysis Metrics

    print("Saving Phase Scoring Metrics:")
    with open('../outputs/metrics/Right_Ground_Contact.txt', 'w') as f:
        f.write(
            f"\n\n-------------------------------------------------------------------------------------------------------------\n\n"

          f"Right Ground Contact:\n\n"

          f"Phase Scores:\n"
          f"{phase_scores[rgc_phase_index][:, 0]}\n\n"

          f"Faulty Reps:\n"
          f"{rep_index[rgc_phase_index][faulty_phases[rgc_phase_index]]}\n\n"

          f"Faulty Rep Frames:\n"
          f"{faulty_phase_frames[0]}"

          f"\n\n-------------------------------------------------------------------------------------------------------------\n\n"
        )
    print(f"{'Right_Ground_Contact.txt':<25} Successfully Saved")

    with open('../outputs/metrics/Right_Propulsion.txt', 'w') as f:
        f.write(
            f"\n\n-------------------------------------------------------------------------------------------------------------\n\n"

            f"Right Propulsion:\n\n"

            f"Phase Scores:\n"
            f"{phase_scores[rp_phase_index][:, 0]}\n\n"

            f"Faulty Reps:\n"
            f"{rep_index[rp_phase_index][faulty_phases[rp_phase_index]]}\n\n"

            f"Faulty Rep Frames:\n"
            f"{faulty_phase_frames[1]}"

            f"\n\n-------------------------------------------------------------------------------------------------------------\n\n"
        )
    print(f"{'Right_Propulsion.txt':<25} Successfully Saved")

    with open('../outputs/metrics/Right_Flight.txt', 'w') as f:
        f.write(
            f"\n\n-------------------------------------------------------------------------------------------------------------\n\n"

            f"Right Flight:\n\n"

            f"Phase Scores:\n"
            f"{phase_scores[rf_phase_index][:, 0]}\n\n"

            f"Faulty Reps:\n"
            f"{rep_index[rf_phase_index][faulty_phases[rf_phase_index]]}\n\n"

            f"Faulty Rep Frames:\n"
            f"{faulty_phase_frames[2]}"

            f"\n\n-------------------------------------------------------------------------------------------------------------\n\n"
        )
    print(f"{'Right_Flight.txt':<25} Successfully Saved")

    with open('../outputs/metrics/Left_Ground_Contact.txt', 'w') as f:
        f.write(
            f"\n\n-------------------------------------------------------------------------------------------------------------\n\n"

            f"Left Ground Contact:\n\n"

            f"Phase Scores:\n"
            f"{phase_scores[lgc_phase_index][:, 0]}\n\n"

            f"Faulty Reps:\n"
            f"{rep_index[lgc_phase_index][faulty_phases[lgc_phase_index]]}\n\n"

            f"Faulty Rep Frames:\n"
            f"{faulty_phase_frames[3]}"

            f"\n\n-------------------------------------------------------------------------------------------------------------\n\n"
        )
    print(f"{'Left_Ground_Contact.txt':<25} Successfully Saved")

    with open('../outputs/metrics/Left_Propulsion.txt', 'w') as f:
        f.write(
            f"\n\n-------------------------------------------------------------------------------------------------------------\n\n"

          f"Left Propulsion:\n\n"

          f"Phase Scores:\n"
          f"{phase_scores[lp_phase_index][:, 0]}\n\n"

          f"Faulty Reps:\n"
          f"{rep_index[lp_phase_index][faulty_phases[lp_phase_index]]}\n\n"

          f"Faulty Rep Frames:\n"
          f"{faulty_phase_frames[4]}"

          f"\n\n-------------------------------------------------------------------------------------------------------------\n\n"
        )
    print(f"{'Left_Propulsion.txt':<25} Successfully Saved")

    with open('../outputs/metrics/Left_Flight.txt', 'w') as f:
        f.write(
            f"\n\n-------------------------------------------------------------------------------------------------------------\n\n"

          f"Left Flight:\n\n"

          f"Phase Scores:"
          f"\n{phase_scores[lf_phase_index][:, 0]}\n\n"

          f"Faulty Reps:\n"
          f"{rep_index[lf_phase_index][faulty_phases[lf_phase_index]]}\n\n"

          f"Faulty Rep Frames:"
          f"\n{faulty_phase_frames[5]}"

          f"\n\n-------------------------------------------------------------------------------------------------------------"
        )
    print(f"{'Left_Flight.txt':<25} Successfully Saved\n")

    # Save Ground Contact Timing
    print("Saving Ground Contact Timing Data\n")
    with open('../outputs/metrics/Ground_Contact_Timing.txt', 'w') as f:
        f.write(
            f"""
    ----------------
    STRIDE FREQUENCY
    ----------------

{steps_per_minute:.3f} Steps per Minute

    ---------------------
    GROUND STRIKE POINTS
    ---------------------

Average Right Ground Strike Point:
{np.mean(right_contacts):.3f}

Average Left Ground Strike Point:
{np.mean(left_contacts):.3f}

Average Strike Point Imbalance: (negative = left | positive = right)
{np.mean(right_contacts) - np.mean(left_contacts):.3f}


Right Ground Striking Points:
{right_contacts}

Left Ground Striking Points:
{left_contacts}

    --------------------
    GROUND CONTACT TIMES
    --------------------

Average Ground Contact Time:
Frames: {(right_on_ground + left_on_ground) / rep_index[-1]:.0f}
Seconds: {(right_on_ground + left_on_ground) / (rep_index[-1] * 30):.3f}


Average Right Ground Contact Time:
Frames: {right_on_ground / (rep_index[-1] // 2):.0f}
Seconds: {right_on_ground / ((rep_index[-1] * 30) // 2):.3f}

Right Ground Contact Time:
Frames: {right_contact_lengths}
Seconds: {right_contact_lengths / 30}


Average Left Ground Contact Time:
Frames: {left_on_ground / (rep_index[-1] // 2):.0f}
Seconds: {left_on_ground / ((rep_index[-1] * 30) // 2):.3f}

Left Ground Contact Time:
Frames: {left_contact_lengths}
Seconds: {left_contact_lengths / 30}
""")
    print(f"{'Ground_Contact_Timing.txt':<25} Successfully Saved\n")

    # Save Graphs of each features scores over time
    print("Saving Z-score Graphs\n")
    plt.figure(figsize=(18, 5))
    plt.title('Phase Z-Scores')
    plt.xlabel("Phase #")
    plt.ylabel("Z-Score")
    plt.plot(phase_scores[:, 0], color='b', label='Score')
    plt.savefig("../outputs/graphs/phase_breakdown/Total_Z-Score.png", dpi=300)
    plt.close()

    if graph_saving:
        for i in range(len(scored_data)):
            plt.figure(figsize=(18, 5))
            plt.title(FEATURE_STRINGS[i])
            plt.xlabel("Frames")
            plt.ylabel("Z Scores")
            plt.axhline(0, color='black', linewidth=2)

            plt.plot(scored_data[i], alpha =.25, color='b', label="Score")

            ema = [scored_data[i, 0]]
            for x in range(1, len(scored_data[i])):
                ema.append(scored_data[i, x] * .05 + ema[-1] * .95 )

            plt.plot(ema, color='r', label="EMA Trend")

            plt.legend()
            plt.savefig(f"../outputs/graphs/Z-Scores/{FEATURE_STRINGS[i]}.png", dpi=300)
            plt.close()

    print("Saving Stride Frequency Data:")
    plt.figure(figsize=(18, 5))
    plt.plot(strides_per_sec, label="Strides per Second")
    plt.title("Stride Frequency")
    plt.ylim(0, None)
    plt.xlabel("Frames")
    plt.ylabel("Strides/Second")
    plt.savefig("../outputs/graphs/Stride_Frequency/Stride_Frequency.png", dpi=300)
    print(f"{'Stride Frequency Data':<25} Successfully Saved\n")

    print('Saving Phase Breakdown Data:')
    phase_score_fig = go.Figure()
    phase_score_fig.add_trace(go.Scatterpolar(
        r=[rgc_breakdown, rp_breakdown, rf_breakdown, lgc_breakdown, lp_breakdown, lf_breakdown, rgc_breakdown],
        theta=['Right Ground Contact', 'Right Propulsion', 'Right Flight',
              'Left Ground Contact', 'Left Propulsion', 'Left Flight', 'Right Ground Contact'],
        fill='toself',
        name='Phase Breakdown',
        line_color='red'
    ))
    # phase_score_fig.write_html('../outputs/graphs/phase_averages.html', auto_open=False) # <-- USE HTML FOR FINAL PRODUCT
    phase_score_fig.write_image('../outputs/graphs/phase_breakdown/phase_breakdown_comparison.png', width=800, height=600, scale=2)
    print(f"{'Phase Breakdown Data':<25} Successfully Saved\n")

    print("Saving Phase Overlay Videos:")
    save_path = '../outputs/videos/overlays/'

    # Save user_input of each phase's best instance overlaid on the worst instance
    save_video(save_path + 'Right_Ground_Contact.mp4',
               phase_scores[rgc_phase_index][:, 1][np.argmax(phase_scores[rgc_phase_index][:, 0])],
               phase_lengths[rgc_phase_index][np.argmax(phase_scores[rgc_phase_index][:, 0])],
               phase_scores[rgc_phase_index][:, 1][np.argmin(phase_scores[rgc_phase_index][:, 0])],
               phase_lengths[rgc_phase_index][np.argmin(phase_scores[rgc_phase_index][:, 0])],
               raw_data, user_predictions, scored_data, FEATURE_STRINGS)

    save_video(save_path + 'Right_Propulsion.mp4',
               phase_scores[rp_phase_index][:, 1][np.argmax(phase_scores[rp_phase_index][:, 0])],
               phase_lengths[rp_phase_index][np.argmax(phase_scores[rp_phase_index][:, 0])],
               phase_scores[rp_phase_index][:, 1][np.argmin(phase_scores[rp_phase_index][:, 0])],
               phase_lengths[rp_phase_index][np.argmin(phase_scores[rp_phase_index][:, 0])],
               raw_data, user_predictions, scored_data, FEATURE_STRINGS)

    save_video(save_path + 'Right_Flight.mp4',
               phase_scores[rf_phase_index][:, 1][np.argmax(phase_scores[rf_phase_index][:, 0])],
               phase_lengths[rf_phase_index][np.argmax(phase_scores[rf_phase_index][:, 0])],
               phase_scores[rf_phase_index][:, 1][np.argmin(phase_scores[rf_phase_index][:, 0])],
               phase_lengths[rf_phase_index][np.argmin(phase_scores[rf_phase_index][:, 0])],
               raw_data, user_predictions, scored_data, FEATURE_STRINGS)

    save_video(save_path + 'Left_Ground_Contact.mp4',
               phase_scores[lgc_phase_index][:, 1][np.argmax(phase_scores[lgc_phase_index][:, 0])],
               phase_lengths[lgc_phase_index][np.argmax(phase_scores[lgc_phase_index][:, 0])],
               phase_scores[lgc_phase_index][:, 1][np.argmin(phase_scores[lgc_phase_index][:, 0])],
               phase_lengths[lgc_phase_index][np.argmin(phase_scores[lgc_phase_index][:, 0])],
               raw_data, user_predictions, scored_data, FEATURE_STRINGS)

    save_video(save_path + 'Left_Propulsion.mp4',
               phase_scores[lp_phase_index][:, 1][np.argmax(phase_scores[lp_phase_index][:, 0])],
               phase_lengths[lp_phase_index][np.argmax(phase_scores[lp_phase_index][:, 0])],
               phase_scores[lp_phase_index][:, 1][np.argmin(phase_scores[lp_phase_index][:, 0])],
               phase_lengths[lp_phase_index][np.argmin(phase_scores[lp_phase_index][:, 0])],
               raw_data, user_predictions, scored_data, FEATURE_STRINGS)

    save_video(save_path + 'Left_Flight.mp4',
               phase_scores[lf_phase_index][:, 1][np.argmax(phase_scores[lf_phase_index][:, 0])],
               phase_lengths[lf_phase_index][np.argmax(phase_scores[lf_phase_index][:, 0])],
               phase_scores[lf_phase_index][:, 1][np.argmin(phase_scores[lf_phase_index][:, 0])],
               phase_lengths[lf_phase_index][np.argmin(phase_scores[lf_phase_index][:, 0])],
               raw_data, user_predictions, scored_data, FEATURE_STRINGS)

#     DEMO OUTPUT
    PHASE_STRINGS = ['Right Ground Contact',
                     'Right Propulsion',
                     'Right Flight',
                     'Left Ground Contact',
                     'Left Propulsion',
                     'Left Flight']

    feature_colors = {
        'red': [(64, 64, 255), (96, 96, 255), (128, 128, 255), (160, 160, 255), (192, 192, 255)],
        'yellow': [(64, 255, 255), (96, 255, 255), (128, 255, 255), (160, 255, 255), (192, 255, 255)],
        'green': [(64, 255, 64), (96, 255, 96), (128, 255, 128), (160, 255, 160), (192, 255, 192)]
    }

    overlay_cap = cv2.VideoCapture(f'../outputs/videos/overlays/full_overlay.mp4')
    skeleton_cap = cv2.VideoCapture(f'../outputs/videos/user_skeleton/user_skeleton.mp4')
    width = int(overlay_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(overlay_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    overlay_cap.set(cv2.CAP_PROP_POS_FRAMES, Engine.get_formatting()[1])
    skeleton_cap.set(cv2.CAP_PROP_POS_FRAMES, Engine.get_formatting()[1])

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    demo = cv2.VideoWriter('../outputs/videos/dashboard/dashboard.mp4', fourcc, 10, (int(width*.65), int(height*.65)))
    canvas = np.zeros((height, width*2, 3), dtype=np.uint8)

    plt.style.use('dark_background')  # Looks better in tech demos
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=100)

    graph = FigureCanvasAgg(fig)
    x_data = np.arange(30)  # Showing a 30-frame window
    line, = ax.plot(x_data, np.zeros(30), color='b', lw=3)

    frame_scores = np.sum(scored_data ** 2, axis=0)
    y_max = max(frame_scores)
    y_green = int(np.percentile(frame_scores, 80))
    y_yellow = int(np.percentile(frame_scores, 95))

    green_zone = range(y_green)
    yellow_zone = range(y_green, y_yellow)

    ax.set_ylim(0, y_max)
    ax.axhspan(0, y_green, facecolor=(0.71, 0.84, 0.66), zorder=0)  # Bottom 1/3 (Safe)
    ax.axhspan(y_green, y_yellow, facecolor=(1.0, 0.9, 0.6), zorder=0)  # Middle 1/3 (Warning)
    ax.axhspan(y_yellow, y_max, facecolor=(0.92, 0.6, 0.6), zorder=0)

    fps = 30

    def phase_to_seconds(x, pos):
        if x in range(len(user_predictions)):
            return f'{x / fps:.2f}s | {phase_strings[user_predictions[int(x)]].upper()}'
        else:
            return f'{x / fps:.2f}s'

    # Set up the formatter on your existing ax
    ax.xaxis.set_major_formatter(FuncFormatter(phase_to_seconds))
    ax.set_xlabel("Time (Seconds) | Phase", fontsize=14, fontweight='bold')
    ax.set_ylabel("Total Form Deviation - Σ(MAD^2)", fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    lih = []
    for frame, (pred1, pred2) in enumerate(zip(user_predictions, user_predictions[1:])):
        if pred1 != pred2:
            lih.append(frame)
        else:
            lih.append(None)

    for frame in range(len(user_predictions)):
        ret1, overlay_frame = overlay_cap.read()
        ret2, skeleton_frame = skeleton_cap.read()

        frame_score = int(np.sum(scored_data[:, frame] ** 2, axis=0))

        if frame_score in green_zone:
            zone = 'green'
            skeleton_frame = np.clip(skeleton_frame * [0.5, 1, 0.5], 0, 255).astype(np.uint8)
        elif frame_score in yellow_zone:
            zone = 'yellow'
            skeleton_frame = np.clip(skeleton_frame * [0.5, 1, 1], 0, 255).astype(np.uint8)
        else:
            zone = 'red'
            skeleton_frame = np.clip(skeleton_frame * [0.5, 0.5, 1], 0, 255).astype(np.uint8)

        top_half = cv2.hconcat([overlay_frame, skeleton_frame])
        full_screen = cv2.vconcat([top_half, canvas])

        cv2.putText(full_screen, 'Current Phase:',
                    (int(width * 2 * .025), int(height * 2 * .5275)), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2,
                    cv2.LINE_AA)

        cv2.putText(full_screen, PHASE_STRINGS[user_predictions[frame]],
                    (int(width*2 * .025), int(height*2 * .5925)), cv2.FONT_HERSHEY_SIMPLEX, 6, (255,255,255), 15, cv2.LINE_AA)

        start = max(0, frame-30)
        scores = list(np.sum(scored_data[:, start+1:frame+1] ** 2, axis=0))
        scores = [0 for _ in range(30 - len(scores))] + scores

        line.set_ydata(scores)
        line.set_xdata(np.arange(frame-29, frame+1))
        ax.set_xlim(frame - 30, frame+1)

        for line_obj in ax.get_lines()[1:]:  # Keep the first line (the data plot)
            line_obj.remove()

        piece = [0 for _ in range(30 - len(lih[start:frame]))] + lih[start:frame]
        for coord in piece:
            if coord:
                ax.axvline(coord, linewidth=0.75, color='black')

        ax.axvline(frame, linewidth=2, color='r')

        graph.draw()
        rgba_buffer = graph.buffer_rgba()
        graph_img = np.asarray(rgba_buffer)
        graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)

        graph_ratio = width / graph_img.shape[1]
        graph_img = cv2.resize(graph_img, (int(width*1.5), int(graph_img.shape[0] * graph_ratio * 1.5)), interpolation=cv2.INTER_AREA)

        h, w = graph_img.shape[:2]
        full_screen[int(height*1.25):int(height*1.25) + h, :w] = graph_img

        error_index = np.argsort(np.abs(scored_data[:, frame]))[-5:][::-1]

        cv2.putText(full_screen,
                    'Greatest Feature Errors:',
                    (int(width*2 * .71), int(height * 1.35)), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (255,255,255), 2, cv2.LINE_AA)

        sign = '(+)' if scored_data[error_index[0], frame] >= 0 else '(-)'

        cv2.putText(full_screen,
                    f'1. {FEATURE_STRINGS[error_index[0]]} {sign}',
                    (int(width*2 * .71), int(height * 1.425)), cv2.FONT_HERSHEY_SIMPLEX, 1.25, feature_colors[zone][0], 2, cv2.LINE_AA)

        sign = '(+)' if scored_data[error_index[1], frame] >= 0 else '(-)'

        cv2.putText(full_screen,
                    f'2. {FEATURE_STRINGS[error_index[1]]} {sign}',
                    (int(width*2 * .71), int(height * 1.475)), cv2.FONT_HERSHEY_SIMPLEX, 1.25, feature_colors[zone][1], 2, cv2.LINE_AA)

        sign = '(+)' if scored_data[error_index[2], frame] >= 0 else '(-)'

        cv2.putText(full_screen,
                    f'3. {FEATURE_STRINGS[error_index[2]]} {sign}',
                    (int(width*2 * .71), int(height * 1.525)), cv2.FONT_HERSHEY_SIMPLEX, 1.25, feature_colors[zone][2], 2, cv2.LINE_AA)

        sign = '(+)' if scored_data[error_index[3], frame] >= 0 else '(-)'

        cv2.putText(full_screen,
                    f'4. {FEATURE_STRINGS[error_index[3]]} {sign}',
                    (int(width*2 * .71), int(height * 1.575)), cv2.FONT_HERSHEY_SIMPLEX, 1.25, feature_colors[zone][3], 2, cv2.LINE_AA)

        sign = '(+)' if scored_data[error_index[4], frame] >= 0 else '(-)'

        cv2.putText(full_screen,
                    f'5. {FEATURE_STRINGS[error_index[4]]} {sign}',
                    (int(width*2 * .71), int(height * 1.625)), cv2.FONT_HERSHEY_SIMPLEX, 1.25, feature_colors[zone][4], 2, cv2.LINE_AA)

        full_screen = cv2.resize(full_screen, (int(width*.65), int(height*.65)), interpolation=cv2.INTER_AREA)
        demo.write(full_screen)

    demo.release()

main()