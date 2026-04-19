from model.creation import create_model
from model.training import train_model
from pose.video_processor import get_data
from statistics.data_processor import get_phase_statistics
import pickle

import numpy as np # to be removed

def main():
    """
    Creates and trains 1D CNN Model and finds Median Absolute Deviation based on the training data
    """
    training_data, raw_training_data = get_data()
    answer_key = np.load('../data/answerkey/full_answer_key.npy') # To be changed

    create_model()
    train_model(training_data, answer_key)
    train_model(training_data, answer_key)

    phase_statistics = get_phase_statistics(training_data, raw_training_data)
    # phase_statistics[phase][subphase][mad/median]

    print(phase_statistics)
    with open('../assets/phase_statistics.pkl', 'wb') as f:
        pickle.dump(phase_statistics, f)

main()