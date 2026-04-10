import numpy as np
from tensorflow import keras

BOE1 = 'boeTreadpt1_capture.npy'
BOE2 = 'boeTreadpt2_capture.npy'
reference_data = np.load(BOE1)
raw_data = np.load('../../raw_data.npy') # [frame, feature]
answer_key = np.load('../../answer_key.npy')
model = keras.models.load_model('../../stride_model_50.keras')

def find_MAD(input):
    """
    Finds the Median Absolute Deviation of the inputted array
    Args:
        input: array of values to be used
    Returns:
        tuple - z scores of all values, MAD of each feature, median of each feature
    """
    median = np.median(input, axis=1)[:, None]
    deviation = np.abs(input - median)
    MAD = np.maximum(np.median(deviation, axis=1), 1e-7)[:, None]  # ensure all values are non-zero
    return MAD, median


reference_predictions = np.argmax(model.predict(reference_data), axis=1)

incorrect = 0
for out, ans in zip(reference_predictions, answer_key):
    # print(out,ans)
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

rgc_MAD, rgc_median = find_MAD(rgc)
rp_MAD,  rp_median  = find_MAD(rp)
rf_MAD,  rf_median  = find_MAD(rf)
lgc_MAD, lgc_median = find_MAD(lgc)
lp_MAD,  lp_median  = find_MAD(lp)
lf_MAD,  lf_median  = find_MAD(lf)

export = np.array([[rgc_MAD, rgc_median],
                   [rp_MAD,  rp_median],
                   [rf_MAD,  rf_median],
                   [lgc_MAD, lgc_median],
                   [lp_MAD,  lp_median],
                   [lf_MAD,  lf_median]])
print(export)
print(export.shape)

np.save("../../phase_scoring.npy", export)

