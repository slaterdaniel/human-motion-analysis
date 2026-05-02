import numpy as np
from tensorflow.keras.models import load_model
from pose import video_processor


def interpolate_phase(phase, phase_num, reference_predictions, n_interp=9):
    """
    Interpolate the given phase to normalize the length of each phase to the same number of frames
    Args:
        phase: NumPy array - phase to be normalized
        phase_num: int - value corresponding to the phase in reference_predictions
        reference_predictions: NumPy array - 1D CNN model phase predictions for each frame
        n_interp: int - number of frames to normalize each phase to
    Returns:
        NumPy array phase array
    """
    last_prediction = 6
    phase_count = 0
    for prediction in reference_predictions:
        if prediction == phase_num and last_prediction != phase_num:
            phase_count += 1
        last_prediction = prediction

    last = False
    start = 0
    progress = 0
    count = 0
    new_phase = np.zeros((phase_count, len(phase), n_interp))
    for current in np.concatenate([reference_predictions == phase_num, [False]]):
        if current:
            if not last:
                start = progress
            progress += 1

        elif last and not current:
            frames = [start + x for x in range(progress - start)]
            step = (progress - start - 1) / (n_interp - 1)
            missing = [start + (step * x) for x in range(n_interp)]
            for i in range(len(phase)):
                y = phase[i, start:progress]
                new = np.interp(missing, frames, y)
                new_phase[count, i] = new
            count += 1

        last = current
    return new_phase


def find_MAD(phase):
    # KEY | Subphases:
    # median0 = early phase
    # median1 = middle phase
    # median2 = late phase

    # Medians of each subphase
    median0 = np.median(phase[:, :, :3], axis=(0, 2))[:, None]
    median1 = np.median(phase[:, :, 3:6], axis=(0, 2))[:, None]
    median2 = np.median(phase[:, :, 6:], axis=(0, 2))[:, None]

    # Deviations of each subphase
    deviation0 = np.abs(phase[:, :, :3] - median0)
    deviation1 = np.abs(phase[:, :, 3:6] - median1)
    deviation2 = np.abs(phase[:, :, 6:] - median2)

    # Median Absolute Deviation of each subphase
    mad0 = np.maximum(np.median(deviation0, axis=(0,2)), 1e-7)[:, None]
    mad1 = np.maximum(np.median(deviation1, axis=(0,2)), 1e-7)[:, None]
    mad2 = np.maximum(np.median(deviation2, axis=(0,2)), 1e-7)[:, None]

    # Save phases into "early", "middle", and "late" subphases
    phase_stats = {
        "early": {
            "mad": mad0,
            "median": median0
        },
        "middle": {
            "mad": mad1,
            "median": median1
        },
        "late": {
            "mad": mad2,
            "median": median2
        }
    }

    return phase_stats


def get_phase_statistics(reference_data, ref_raw_data):
    """
    Main function
    Args:
        reference data - NumPy array formatted for 1D CNN 9 frame windows
        raw data - NumPy array where shape = [feature, frame]
    Returns:
        tuple - Median Absolute Deviations and medians of each phase split into 3 subphases of 3 frames each
    """

    model = load_model('../assets/phase_classifier50.keras', compile=False)
    reference_predictions = np.argmax(model.predict(reference_data), axis=1)

    # save raw data by phase while being grouped by feature
    rgc = ref_raw_data[reference_predictions == 0].T
    rp = ref_raw_data[reference_predictions == 1].T
    rf = ref_raw_data[reference_predictions == 2].T
    lgc = ref_raw_data[reference_predictions == 3].T
    lp = ref_raw_data[reference_predictions == 4].T
    lf = ref_raw_data[reference_predictions == 5].T

    new_rgc = interpolate_phase(rgc, 0, reference_predictions)
    new_rp = interpolate_phase(rp, 1, reference_predictions)
    new_rf = interpolate_phase(rf, 2, reference_predictions)
    new_lgc = interpolate_phase(lgc, 3, reference_predictions)
    new_lp = interpolate_phase(lp, 4, reference_predictions)
    new_lf = interpolate_phase(lf, 5, reference_predictions)

    rgc_stats = find_MAD(new_rgc)
    rp_stats = find_MAD(new_rp)
    rf_stats = find_MAD(new_rf)
    lgc_stats = find_MAD(new_lgc)
    lp_stats = find_MAD(new_lp)
    lf_stats = find_MAD(new_lf)

    phase_stats = {
        "rgc": rgc_stats,
        "rp": rp_stats,
        "rf": rf_stats,
        "lgc": lgc_stats,
        "lp": lp_stats,
        "lf": lf_stats
    }

    return phase_stats

