from model.creation import create_models
from model.training import train_models
from model.answer_key import create_key
from statistics.data_processor import get_phase_statistics
import pickle

def main():
    """
    Creates and trains 1D CNN Model and finds Median Absolute Deviation based on the training data
    """
    while True:
        all_engines = ['mediapipe', 'yolo26', 'mmpose']

        to_create = input("Enter Engines to create:\n").lower()
        if to_create == 'all':
            engines = all_engines
            break

        engines = to_create.split()
        if set(engines) - set(all_engines):
            print(f'Invalid Engine Inputted. Please Try Again.\n')
            continue
        break

    show_process = input('\nDo You Want the Skeleton Overlays to be shown during processing time? This may so down processing by ~20%\n[y/n]\n').lower()
    show = True if show_process[0] == 'y' else False

    training_data = []
    raw_data = []

    for name in engines:
        if name == 'mediapipe':
            from pose import mediapipe_video_processor
            training, raw = mediapipe_video_processor.get_data(show)
            training_data.append(training)
            raw_data.append(raw)

        elif name == 'yolo26':
            from pose import yolo26_video_processor
            training, raw = yolo26_video_processor.get_data(show)
            training_data.append(training)
            raw_data.append(raw)

        elif name == 'mmpose':
            from pose import mmpose_video_processor
            training, raw = mmpose_video_processor.get_data(show)
            training_data.append(training)
            raw_data.append(raw)

    answer_key = create_key()

    create_models(engines)
    train_models(training_data, answer_key, engines)
    train_models(training_data, answer_key, engines)

    new_phase_stats = get_phase_statistics(training_data, raw_data, engines)
    # new_phase_stats[engine][phase][subphase][mad/median]

    print(new_phase_stats)
    with open('../assets/phase_statistics.pkl', 'r+b') as f:
        phase_statistics = pickle.load(f)
        phase_statistics.update(new_phase_stats)

        f.seek(0)
        pickle.dump(phase_statistics, f)
        f.truncate()

main()