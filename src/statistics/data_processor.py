import numpy as np
from tensorflow.keras.models import load_model


def phase_interpolation(phase, phase_num, reference_predictions, n_interp=9):
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
    # Medians of each third
    median1 = np.median(phase[:, :, :3], axis=(0, 2))[:, None]
    median2 = np.median(phase[:, :, 3:6], axis=(0, 2))[:, None]
    median3 = np.median(phase[:, :, 6:], axis=(0, 2))[:, None]

    # Deviations of each third
    deviation1 = np.abs(phase[:, :, :3] - median1)
    deviation2 = np.abs(phase[:, :, 3:6] - median2)
    deviation3 = np.abs(phase[:, :, 6:] - median3)

    # Median Absolute Deviation of each phase
    mad1 = np.maximum(np.median(deviation1, axis=1), 1e-7)[:, None]
    mad2 = np.maximum(np.median(deviation2, axis=1), 1e-7)[:, None]
    mad3 = np.maximum(np.median(deviation3, axis=1), 1e-7)[:, None]

    return mad1, median1, mad2, median2, mad3, median3


def get_phase_statistics():
    """
    Main function
    Returns:
        tuple - Median Absolute Deviations and medians of each phase split into 3 subphases of 3 frames each
    """
    reference_data = np.load("../../reference_capture.npy")
    raw_data = np.load('../../raw_data.npy')  # [frame, feature]
    model = load_model('../../stride_model_50.keras')
    reference_predictions = np.argmax(model.predict(reference_data), axis=1)

    # save raw data by phase while being grouped by feature
    rgc = raw_data[reference_predictions == 0].T
    rp = raw_data[reference_predictions == 1].T
    rf = raw_data[reference_predictions == 2].T
    lgc = raw_data[reference_predictions == 3].T
    lp = raw_data[reference_predictions == 4].T
    lf = raw_data[reference_predictions == 5].T

    new_rgc = phase_interpolation(rgc, 0, reference_predictions)
    new_rp = phase_interpolation(rp, 1, reference_predictions)
    new_rf = phase_interpolation(rf, 2, reference_predictions)
    new_lgc = phase_interpolation(lgc, 3, reference_predictions)
    new_lp = phase_interpolation(lp, 4, reference_predictions)
    new_lf = phase_interpolation(lf, 5, reference_predictions)

    rgc_stats = find_MAD(new_rgc)
    rp_stats = find_MAD(new_rp)
    rf_stats = find_MAD(new_rf)
    lgc_stats = find_MAD(new_lgc)
    lp_stats = find_MAD(new_lp)
    lf_stats = find_MAD(new_lf)

    return rgc_stats, rp_stats, rf_stats, lgc_stats, lp_stats, lp_stats, lf_stats

