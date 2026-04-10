import cv2
import numpy as np
from tensorflow import keras

# KEY:
# rgc = right ground contact
# rp  = right propulsion phase
# rf  = right flight phase
# lgc = left ground contact
# lp  = left propulsion phase
# lf  = left flight phase

def find_phase_scores(data, user_predictions, window_size=9):
    """
    Finds phase scores of the users data
    Args:
        data: numpy array of features z scores where: rows = features, columns = frame
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
        score = np.sum(data[:, progress:progress + length] ** 2)
        total_scores.append((score / length, progress + (window_size//2)))
        progress += length

    return np.array(total_scores), np.array(phase_lengths)

def save_video(file, worst_frame, worst_length, best_frame, best_length, raw_data, user_predictions, scored_data, window_size=9):
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
    cap = cv2.VideoCapture('../../../user_skeleton.mp4')
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

        # Final 30 = mediapipe landmark coords
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
        if not ret:
            break
        if i < offset or i >= offset + length:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

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
    for i in range(length + (offset * 2)):
        ret, read_frame = cap.read()
        if not ret:
            break
        read_frame = np.clip(read_frame * [0.5, 1, 0.5], 0, 255).astype(np.uint8)

        blended = cv2.addWeighted(worst[i], 1, read_frame, 1, 0)
        for _ in range(12): out.write(blended)
    out.release()
    print(f"{file:<25} Successfully Saved")

def main():
    user_data = np.load('../../../user_capture.npy') # user data formatted for 1D CNN 9 frame windows
    raw_data = np.load('../../../user_raw_data.npy').T # raw user data where shape = [feature, frame]
    phase_scoring = np.load('../../../phase_scoring.npy') # Mean Absolute Deviation (MAD) of reference data
    scored_data = np.zeros(raw_data.shape) # initialize empty numpy array for storing scored user data

    model = keras.models.load_model('../../../stride_model_50.keras') # 1D CNN to predict phases
    user_predictions = np.argmax(model.predict(user_data), axis=1) # predict the phases from users data

    np.set_printoptions(suppress=True, precision=4, linewidth=95)

    # Load Mean Absolute Deviation from training
    rgc_mad, rgc_median = phase_scoring[0]
    rp_mad, rp_median   = phase_scoring[1]
    rf_mad, rf_median   = phase_scoring[2]
    lgc_mad, lgc_median = phase_scoring[3]
    lp_mad, lp_median   = phase_scoring[4]
    lf_mad, lf_median   = phase_scoring[5]

    # Save Phase Indexes
    rgc_index = user_predictions == 0
    rp_index  = user_predictions == 1
    rf_index  = user_predictions == 2
    lgc_index = user_predictions == 3
    lp_index  = user_predictions == 4
    lf_index  = user_predictions == 5

    # Z scores from MAD
    scored_data[:, rgc_index] = 0.6745 * np.abs(raw_data[:, rgc_index] - rgc_median) / rgc_mad
    scored_data[:, rp_index]  = 0.6745 * np.abs(raw_data[:, rp_index]  - rp_median)  / rp_mad
    scored_data[:, rf_index]  = 0.6745 * np.abs(raw_data[:, rf_index]  - rf_median)  / rf_mad
    scored_data[:, lgc_index] = 0.6745 * np.abs(raw_data[:, lgc_index] - lgc_median) / lgc_mad
    scored_data[:, lp_index]  = 0.6745 * np.abs(raw_data[:, lp_index]  - lp_median)  / lp_mad
    scored_data[:, lf_index]  = 0.6745 * np.abs(raw_data[:, lf_index]  - lf_median)  / lf_mad
    print(scored_data.shape)

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

    # Create array of the indexes of each rep as they appear in the user's video
    rep_index = []
    curr = 1
    for i in range(len(phase_order)):
        if phase_order[i] == 0 or phase_order[i] == 3:
            curr += 1
        rep_index.append(curr)
    rep_index = np.array(rep_index)

    # find the indexes of each phase
    rgc_phase_index = phase_order == 0
    rp_phase_index  = phase_order == 1
    rf_phase_index  = phase_order == 2
    lgc_phase_index = phase_order == 3
    lp_phase_index  = phase_order == 4
    lf_phase_index  = phase_order == 5

    # phases are deemed "faulty" when the mean of the sum of the features squared z scores
    # in a given phase are in higher than the 95th percentile of phase z scores
    cutoff = np.percentile(phase_scores[:,0], 95)
    faulty_phases = phase_scores[:, 0] > cutoff

    # Create a dictionary of the starting frames of each "faulty" phase
    faulty_phase_frames = {key: [] for key in range(6)}
    for i in range(len(faulty_phases)):
        if faulty_phases[i]:
            faulty_phase_frames[phase_order[i]].append(int(phase_scores[i, 1]))

    # Output Results
    print(f"\n\n-------------------------------------------------------------------------------------------------------------\n\n"
          
          f"Cutoff: {cutoff}"
          
          f"\n\n-------------------------------------------------------------------------------------------------------------\n\n"
          
          f"Right Ground Contact:\n\n"
          
          f"Phase Scores:\n"
          f"{phase_scores[rgc_phase_index][:, 0]}\n\n"
          
          f"Faulty Reps:\n"
          f"{rep_index[rgc_phase_index][faulty_phases[rgc_phase_index]]}\n\n"
          
          f"Faulty Rep Frames:\n"
          f"{faulty_phase_frames[0]}"
    
          f"\n\n-------------------------------------------------------------------------------------------------------------\n\n"
    
          f"Right Propulsion:\n\n"
          
          f"Phase Scores:\n"
          f"{phase_scores[rp_phase_index][:, 0]}\n\n"
          
          f"Faulty Reps:\n"
          f"{rep_index[rp_phase_index][faulty_phases[rp_phase_index]]}\n\n"
          
          f"Faulty Rep Frames:\n"
          f"{faulty_phase_frames[1]}"
    
          f"\n\n-------------------------------------------------------------------------------------------------------------\n\n"
    
          f"Right Flight:\n\n"
          
          f"Phase Scores:\n"
          f"{phase_scores[rf_phase_index][:, 0]}\n\n"
          
          f"Faulty Reps:\n"
          f"{rep_index[rf_phase_index][faulty_phases[rf_phase_index]]}\n\n"
          
          f"Faulty Rep Frames:\n"
          f"{faulty_phase_frames[2]}"
    
          f"\n\n-------------------------------------------------------------------------------------------------------------\n\n"
    
          f"Left Ground Contact:\n\n"
          
          f"Phase Scores:\n"
          f"{phase_scores[lgc_phase_index][:, 0]}\n\n"
          
          f"Faulty Reps:\n"
          f"{rep_index[lgc_phase_index][faulty_phases[lgc_phase_index]]}\n\n"
          
          f"Faulty Rep Frames:\n"
          f"{faulty_phase_frames[3]}"
    
          f"\n\n-------------------------------------------------------------------------------------------------------------\n\n"
    
          f"Left Propulsion:\n\n"
          
          f"Phase Scores:\n"
          f"{phase_scores[lp_phase_index][:, 0]}\n\n"
          
          f"Faulty Reps:\n"
          f"{rep_index[lp_phase_index][faulty_phases[lp_phase_index]]}\n\n"
          
          f"Faulty Rep Frames:\n"
          f"{faulty_phase_frames[4]}"
    
          f"\n\n-------------------------------------------------------------------------------------------------------------\n\n"
    
          f"Left Flight:\n\n"
          
          f"Phase Scores:"
          f"\n{phase_scores[lf_phase_index][:, 0]}\n\n"
          
          f"Faulty Reps:\n"
          f"{rep_index[lf_phase_index][faulty_phases[lf_phase_index]]}\n\n"
          
          f"Faulty Rep Frames:"
          f"\n{faulty_phase_frames[5]}"
    
          f"\n\n-------------------------------------------------------------------------------------------------------------")

    # Save videos of each phase's best instance overlaid on the worst instance
    save_video('../../../Right_Ground_Contact.mp4',
               phase_scores[rgc_phase_index][:, 1][np.argmax(phase_scores[rgc_phase_index][:, 0])],
               phase_lengths[rgc_phase_index][np.argmax(phase_scores[rgc_phase_index][:, 0])],
               phase_scores[rgc_phase_index][:, 1][np.argmin(phase_scores[rgc_phase_index][:, 0])],
               phase_lengths[rgc_phase_index][np.argmin(phase_scores[rgc_phase_index][:, 0])],
               raw_data, user_predictions, scored_data)

    save_video('../../../Right_Propulsion.mp4',
               phase_scores[rp_phase_index][:, 1][np.argmax(phase_scores[rp_phase_index][:, 0])],
               phase_lengths[rp_phase_index][np.argmax(phase_scores[rp_phase_index][:, 0])],
               phase_scores[rp_phase_index][:, 1][np.argmin(phase_scores[rp_phase_index][:, 0])],
               phase_lengths[rp_phase_index][np.argmin(phase_scores[rp_phase_index][:, 0])],
               raw_data, user_predictions, scored_data)

    save_video('../../../Right_Flight.mp4',
               phase_scores[rf_phase_index][:, 1][np.argmax(phase_scores[rf_phase_index][:, 0])],
               phase_lengths[rf_phase_index][np.argmax(phase_scores[rf_phase_index][:, 0])],
               phase_scores[rf_phase_index][:, 1][np.argmin(phase_scores[rf_phase_index][:, 0])],
               phase_lengths[rf_phase_index][np.argmin(phase_scores[rf_phase_index][:, 0])],
               raw_data, user_predictions, scored_data)

    save_video('../../../Left_Ground_Contact.mp4',
               phase_scores[lgc_phase_index][:, 1][np.argmax(phase_scores[lgc_phase_index][:, 0])],
               phase_lengths[lgc_phase_index][np.argmax(phase_scores[lgc_phase_index][:, 0])],
               phase_scores[lgc_phase_index][:, 1][np.argmin(phase_scores[lgc_phase_index][:, 0])],
               phase_lengths[lgc_phase_index][np.argmin(phase_scores[lgc_phase_index][:, 0])],
               raw_data, user_predictions, scored_data)

    save_video('../../../Left_Propulsion.mp4',
               phase_scores[lp_phase_index][:, 1][np.argmax(phase_scores[lp_phase_index][:, 0])],
               phase_lengths[lp_phase_index][np.argmax(phase_scores[lp_phase_index][:, 0])],
               phase_scores[lp_phase_index][:, 1][np.argmin(phase_scores[lp_phase_index][:, 0])],
               phase_lengths[lp_phase_index][np.argmin(phase_scores[lp_phase_index][:, 0])],
               raw_data, user_predictions, scored_data)

    save_video('../../../Left_Flight.mp4',
               phase_scores[lf_phase_index][:, 1][np.argmax(phase_scores[lf_phase_index][:, 0])],
               phase_lengths[lf_phase_index][np.argmax(phase_scores[lf_phase_index][:, 0])],
               phase_scores[lf_phase_index][:, 1][np.argmin(phase_scores[lf_phase_index][:, 0])],
               phase_lengths[lf_phase_index][np.argmin(phase_scores[lf_phase_index][:, 0])],
               raw_data, user_predictions, scored_data)

main()


# To Do:
# - give sliced videos of phases with error ✅
#       - overlay correct form ✅
#       - use mediapipe skeletons rather than raw video for clarity. ✅
# - FIX PHASE LENGTH FINDER WHEN SAVING VIDEO ✅
# - Try to get the angle velocities to work because they are good for sprinting feedback ✅
# - Create MAD and Median for each part of a phase,
# so the z scores are relative to what part of the phase the user is in. 🔳 --> *** CURRENT ***
# - For the 3 largest mistakes in a phase video --> display the value of the feature nex to where the feature is on the screen. 🔳
# - Detecting sprint phases automatically ✅
# - Measuring stride frequency 🔳
# - Measuring ground contact time 🔳
# - Detecting form breakdown near max speed 🔳
# - Comparing two athletes’ mechanics 🔳
# - Graph scores of each phase over time 🔳
# - Store the scores of the user over time across each time they use the program to track their progress 🔳

# - For Disorders:
# - Automatically detect whether the video is being taken from the side, front, or back 🔳

# After getting more training:
# - provide feedback on what to change and how bad each mistake is (ex. right elbow angle is 10% too large) 🔳
# - return a full length video of the user running while overlaying the correct form the whole time (using skeletons probably) 🔳


# Other data points to output independently:
# - How far foot is landing in front of body 🔳
#       - take x coord of the toe of the first frame of Left/Right Ground Contact phase 🔳
#       - try and do this in meters rather than pixels 🔳


