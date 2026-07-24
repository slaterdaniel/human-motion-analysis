from phase_classes.phase_classes import *
from tensorflow.keras.models import load_model
import numpy as np

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
    print(new_phase.shape)
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


def get_phase_statistics(reference_data, ref_raw_data, engines):
    """
    Main function
    Args:
        reference data - NumPy array formatted for 1D CNN 9 frame windows
        raw data - NumPy array where shape = [feature, frame]
    Returns:
        tuple - Median Absolute Deviations and medians of each phase split into 3 subphases of 3 frames each
    """
    stats = {}

    for data, raw, engine in zip(reference_data, ref_raw_data, engines):
        model = load_model(f'assets/phase_classifier_models/{engine}_phase_classifier.keras', compile=False)
        reference_predictions = np.argmax(model.predict(data), axis=1)

        # save raw data by phase while being grouped by feature
        action = Action(
                        engine,
                        ['rgc', 'rp', 'rf', 'lgc', 'lp', 'lf'],
                        [raw[reference_predictions == 0].T,
                         raw[reference_predictions == 1].T,
                         raw[reference_predictions == 2].T,
                         raw[reference_predictions == 3].T,
                         raw[reference_predictions == 4].T,
                         raw[reference_predictions == 5].T]
        )
        print('engine:', engine)
        print('raw:', raw.shape)
        print('ref:', reference_predictions.shape)

        for i in range(len(action.phases)):
            action.phases[i] = interpolate_phase(action.phases[i], i, reference_predictions)
            action.phase_stats[action.phase_names[i]] = find_MAD(action.phases[i])
            if engine == 'yolo26':
                print('     dict:', i, action.phase_stats[action.phase_names[i]]['early']['median'].shape)

        stats[engine] = action.phase_stats.copy()

    return stats