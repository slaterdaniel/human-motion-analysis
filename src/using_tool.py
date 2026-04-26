from pose import video_processor
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

# KEY:
# rgc = right ground contact
# rp  = right propulsion phase
# rf  = right flight phase
# lgc = left ground contact
# lp  = left propulsion phase
# lf  = left flight phase

def find_phase_scores(phase, user_predictions, window_size=9):
    """
    Finds phase scores of the users data
    Args:
        phase: numpy array of features z scores where: rows = features, columns = frame
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
        score = np.sum(phase[:, progress:progress + length] ** 2)
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
    cap = cv2.VideoCapture('../outputs/videos/user_skeleton/user_skeleton.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    font = cv2.FONT_HERSHEY_SIMPLEX
    style = cv2.LINE_AA

    PHASE_STRINGS = {0: 'Right Ground Contact',
                     1: 'Right Propulsion',
                     2: 'Right Flight',
                     3: 'Left Ground Contact',
                     4: 'Left Propulsion',
                     5: 'Left Flight'}

    border = window_size // 2
    length = np.maximum(worst_length, best_length)

    offset = length // 2 + 1
    worst_start_frame = worst_frame - offset
    best_start_frame = best_frame - offset

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file, fourcc, fps, (width, height))
    worst = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, worst_start_frame)
    for i in range(length + (offset * 2)):
        ret, read_frame = cap.read()
        if not ret or worst_start_frame + i - border >= scored_data.shape[1]:
            break
        if offset <= i < offset + worst_length:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        curr_frame = int(worst_start_frame + i)
        read_frame = np.clip(read_frame * [0.5, 0.5, 1], 0, 255).astype(np.uint8)

        frame_scores = scored_data[:, curr_frame - border]
        score = round(np.sum(frame_scores ** 2), 1)
        biggest_mistake = np.argmax(frame_scores)

        error_value = round(raw_data[biggest_mistake, curr_frame - border], 2)
        reference_value = round(raw_data[biggest_mistake, int(best_start_frame + i) - border], 2)

        cv2.putText(read_frame,
                    f"Phase: {PHASE_STRINGS[user_predictions[curr_frame - border]]}",
                    (int(width * .7), int(height * .45)), font, 1, color, 2, style)

        cv2.putText(read_frame,
                    f"Score: {score}",
                    (int(width * .7), int(height * .5)), font, 1, color, 2, style)

        cv2.putText(read_frame,
                    f"{FEATURE_STRINGS[biggest_mistake]}:",
                    (int(width * .7), int(height * .6)), font, 1, (255, 255, 255), 2, style)

        cv2.putText(read_frame,
                    f"{'Error:':<8} {error_value:<8} Frame: {curr_frame + 1}",
                    (int(width * .7), int(height * .65)), font, 1, (127,127,255), 2, style)

        cv2.putText(read_frame,
                    f"{'Ref:':<8} {reference_value:<8} Frame: {int(best_start_frame + i + 1)}",
                    (int(width * .7), int(height * .7)), font, 1, (127, 255, 127), 2, style)

        cv2.putText(read_frame,
                    f"{'% Error:':<8} {round((error_value - reference_value) * 100 / (reference_value + 1e-8), 2)}%",
                    (int(width * .7), int(height * .75)), font, 1, (127, 255, 255), 2, style)

        worst.append(read_frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, best_start_frame)
    for i in range(len(worst)):
        ret, read_frame = cap.read()
        if not ret:
            break
        read_frame = np.clip(read_frame * [0.5, 1, 0.5], 0, 255).astype(np.uint8)

        blended = cv2.addWeighted(worst[i], 1, read_frame, 1, 0)
        for _ in range(12): out.write(blended)
    cap.release()
    out.release()
    print(f"{file[27:]:<25} Successfully Saved")



def main():

    USER_VIDEO = "../data/videos/user_input/short-boetest.mov" # ***REPLACE WITH FILE OF USER VIDEO***

    np.set_printoptions(threshold=np.inf, suppress=True, precision=3, linewidth=95)

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

    # user data = array formatted for 1D CNN 9 frame windows
    # raw user data = array where shape=[feature, frame]
    user_data, raw_data = video_processor.get_data(USER_VIDEO)
    raw_data = raw_data.T

    # Median Absolute Deviation (MAD) and medians of features in reference data
    with open('../assets/phase_statistics.pkl', 'rb') as f:
        phase_stats = pickle.load(f)

    model = load_model('../assets/phase_classifier50.keras', compile=False) # 1D CNN to predict phases
    user_predictions = np.argmax(model.predict(user_data), axis=1) # predict the phases from users data

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
    right_contacts = raw_data[48, rgc_starts]
    left_contacts = raw_data[46, lgc_starts]

    # Number of frames with right/left foot grounded
    right_on_ground = len(user_predictions[user_predictions == 0]) + len(user_predictions[user_predictions == 1])
    left_on_ground = len(user_predictions[user_predictions == 3]) + len(user_predictions[user_predictions == 4])



    scored_data = np.zeros(raw_data.shape) # initialize empty array for storing scored user data

    phase_strings = {
        0: "rgc",
        1: "rp",
        2: "rf",
        3: "lgc",
        4: "lp",
        5: "lf"
    }
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
                    0.6745 * np.abs(raw_data[:, start:start + base] - phase_stats[phase]["early"]["median"])
                    / phase_stats[phase]["early"]["mad"])

            # "middle" subphase
            scored_data[:, start + base:start + base * 2 + remainder] = (
                    0.6745 * np.abs(raw_data[:, start + base : start + base * 2 + remainder] - phase_stats[phase]["middle"]["median"])
                    / phase_stats[phase]["middle"]["mad"])

            # "late" subphase
            scored_data[:, start + base * 2 + remainder:i] = (
                    0.6745 * np.abs(raw_data[:, start + base * 2 + remainder : start + length] - phase_stats[phase]["late"]["median"])
                    / phase_stats[phase]["late"]["mad"])

            length = 1
        last = current

    # ==============================================================================
    #                            Total Phase Analysis
    # ==============================================================================

    # Score each feature across each frame using MAD Z score
    phase_scores, phase_lengths = find_phase_scores(scored_data, user_predictions)

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
        count = 0

        # count number of frames within window
        for phase in user_predictions[start+1:end]:
            if (phase == 0 or phase == 3) and phase != last_phase: # Count rep when left or right foot hits the ground
                count += 1
            last_phase = phase
        strides_per_sec.append(count * window // (end-start)) # Adjust count to window size

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
{np.mean(right_contacts)}

Average Left Ground Strike Point:
{np.mean(left_contacts)}


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
    print("Saving Z-score Graphs of each feature\n")
    for i in range(len(scored_data)):
        plt.figure(i+1, figsize=(18, 5))
        plt.title(FEATURE_STRINGS[i])
        plt.xlabel("Frames")
        plt.ylabel("Z Scores")

        plt.plot(scored_data[i], alpha =.25, color='b', label="Score")

        ema = [scored_data[i, 0]]
        for x in range(1, len(scored_data[i])):
            ema.append(scored_data[i, x] * .05 + ema[-1] * .95 )

        plt.plot(ema, color='r', label="EMA Trend")

        plt.legend()
        plt.savefig(f"../outputs/graphs/Z-Scores/{FEATURE_STRINGS[i]}.png", dpi=300)
        plt.close()

    print("Saving Stride Frequency Data:")
    plt.figure(0, figsize=(18, 5))
    plt.plot(strides_per_sec, label="Strides per Second")
    plt.title("Stride Frequency")
    plt.xlabel("Frames")
    plt.ylabel("Strides/Second")
    plt.savefig("../outputs/graphs/Stride_Frequency/Stride_Frequency.png", dpi=300)
    print(f"{'Stride Frequency Data':<25} Successfully Saved\n")

    print("Saving Phase Overlay Videos:")

    # Save user_input of each phase's best instance overlaid on the worst instance
    save_video('../outputs/videos/overlay_videos/Right_Ground_Contact.mp4',
               phase_scores[rgc_phase_index][:, 1][np.argmax(phase_scores[rgc_phase_index][:, 0])],
               phase_lengths[rgc_phase_index][np.argmax(phase_scores[rgc_phase_index][:, 0])],
               phase_scores[rgc_phase_index][:, 1][np.argmin(phase_scores[rgc_phase_index][:, 0])],
               phase_lengths[rgc_phase_index][np.argmin(phase_scores[rgc_phase_index][:, 0])],
               raw_data, user_predictions, scored_data, FEATURE_STRINGS)

    save_video('../outputs/videos/overlay_videos/Right_Propulsion.mp4',
               phase_scores[rp_phase_index][:, 1][np.argmax(phase_scores[rp_phase_index][:, 0])],
               phase_lengths[rp_phase_index][np.argmax(phase_scores[rp_phase_index][:, 0])],
               phase_scores[rp_phase_index][:, 1][np.argmin(phase_scores[rp_phase_index][:, 0])],
               phase_lengths[rp_phase_index][np.argmin(phase_scores[rp_phase_index][:, 0])],
               raw_data, user_predictions, scored_data, FEATURE_STRINGS)

    save_video('../outputs/videos/overlay_videos/Right_Flight.mp4',
               phase_scores[rf_phase_index][:, 1][np.argmax(phase_scores[rf_phase_index][:, 0])],
               phase_lengths[rf_phase_index][np.argmax(phase_scores[rf_phase_index][:, 0])],
               phase_scores[rf_phase_index][:, 1][np.argmin(phase_scores[rf_phase_index][:, 0])],
               phase_lengths[rf_phase_index][np.argmin(phase_scores[rf_phase_index][:, 0])],
               raw_data, user_predictions, scored_data, FEATURE_STRINGS)

    save_video('../outputs/videos/overlay_videos/Left_Ground_Contact.mp4',
               phase_scores[lgc_phase_index][:, 1][np.argmax(phase_scores[lgc_phase_index][:, 0])],
               phase_lengths[lgc_phase_index][np.argmax(phase_scores[lgc_phase_index][:, 0])],
               phase_scores[lgc_phase_index][:, 1][np.argmin(phase_scores[lgc_phase_index][:, 0])],
               phase_lengths[lgc_phase_index][np.argmin(phase_scores[lgc_phase_index][:, 0])],
               raw_data, user_predictions, scored_data, FEATURE_STRINGS)

    save_video('../outputs/videos/overlay_videos/Left_Propulsion.mp4',
               phase_scores[lp_phase_index][:, 1][np.argmax(phase_scores[lp_phase_index][:, 0])],
               phase_lengths[lp_phase_index][np.argmax(phase_scores[lp_phase_index][:, 0])],
               phase_scores[lp_phase_index][:, 1][np.argmin(phase_scores[lp_phase_index][:, 0])],
               phase_lengths[lp_phase_index][np.argmin(phase_scores[lp_phase_index][:, 0])],
               raw_data, user_predictions, scored_data, FEATURE_STRINGS)

    save_video('../outputs/videos/overlay_videos/Left_Flight.mp4',
               phase_scores[lf_phase_index][:, 1][np.argmax(phase_scores[lf_phase_index][:, 0])],
               phase_lengths[lf_phase_index][np.argmax(phase_scores[lf_phase_index][:, 0])],
               phase_scores[lf_phase_index][:, 1][np.argmin(phase_scores[lf_phase_index][:, 0])],
               phase_lengths[lf_phase_index][np.argmin(phase_scores[lf_phase_index][:, 0])],
               raw_data, user_predictions, scored_data, FEATURE_STRINGS)

main()