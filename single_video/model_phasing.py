import numpy as np
from tensorflow.keras.models import load_model


BOE1 = 'boeTreadpt1_capture.npy'
BOE2 = 'boeTreadpt2_capture.npy'
reference_data = np.load(BOE1)
raw_data = np.load('../../raw_data.npy') # [frame, feature]
answer_key = np.load('../../answer_key.npy')
model = load_model('../../stride_model_50.keras')

def phase_interpolation(phase, phase_num, reference_predictions):

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
    n_interp = 9
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


def find_MAD(feature_values):
    """
    Finds the Median Absolute Deviation of the inputted array
    Args:
        feature_values: array of values to be used
        predictions: numpy array of the model's predictions of which phase the user is in for each frame
    Returns:
        tuple - z scores of all values, MAD of each feature, median of each feature
    """

    median = np.median(feature_values, axis=1)[:, None]
    deviation = np.abs(feature_values - median)
    mad = np.maximum(np.median(deviation, axis=1), 1e-7)[:, None]  # ensure all values are non-zero
    return mad, median


def main():
    reference_predictions = np.argmax(model.predict(reference_data), axis=1)
    
    incorrect = 0
    for out, ans in zip(reference_predictions, answer_key):
        if out != ans:
            incorrect += 1
    print(f'{len(reference_predictions) - incorrect}/{len(reference_predictions)}')
    
    np.set_printoptions(suppress=True, precision=4, linewidth=95)
    
    # save raw data by phase while being grouped by feature
    print()
    rgc = raw_data[reference_predictions == 0].T
    rp  = raw_data[reference_predictions == 1].T
    rf  = raw_data[reference_predictions == 2].T
    lgc = raw_data[reference_predictions == 3].T
    lp  = raw_data[reference_predictions == 4].T
    lf  = raw_data[reference_predictions == 5].T

    new_rgc = phase_interpolation(rgc, 0, reference_predictions)
    new_rp  = phase_interpolation(rp,  1, reference_predictions)
    new_rf  = phase_interpolation(rf,  2, reference_predictions)
    new_lgc = phase_interpolation(lgc, 3, reference_predictions)
    new_lp  = phase_interpolation(lp,  4, reference_predictions)
    new_lf  = phase_interpolation(lf,  5, reference_predictions)


    third1_med = np.median(new_rgc[:, :, :3], axis=(0,2))[:, None]
    third2_med = np.median(new_rgc[:, :, 3:6], axis=(0,2))[:, None]
    third3_med = np.median(new_rgc[:, :, 6:], axis=(0,2))[:, None]

    deviation1 = np.abs(new_rgc[:, :, :3] - third1_med)

    mad1 = np.maximum(np.median(deviation1, axis=1), 1e-7)[:, None]

    print(third1_med)
    print(third1_med.shape)
    print()
    print(deviation1)
    print(deviation1.shape)
    print()
    print(mad1)
    print(mad1.shape)

    # # to be changed
    rgc_MAD, rgc_median = find_MAD(rgc)

    # rp_MAD,  rp_median  = find_MAD(rp)
    # rf_MAD,  rf_median  = find_MAD(rf)
    # lgc_MAD, lgc_median = find_MAD(lgc)
    # lp_MAD,  lp_median  = find_MAD(lp)
    # lf_MAD,  lf_median  = find_MAD(lf)
    #
    # export = np.array([[rgc_MAD, rgc_median],
    #                    [rp_MAD,  rp_median],
    #                    [rf_MAD,  rf_median],
    #                    [lgc_MAD, lgc_median],
    #                    [lp_MAD,  lp_median],
    #                    [lf_MAD,  lf_median]])
    # print(export)
    # print(export.shape)
    
    # np.save("phase_scoring.npy", export)

main()