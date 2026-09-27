from keras.src.utils.dataset_utils import labels_to_dataset_tf
from tensorflow import keras

# Phases:
# 0 - Right Ground Contact
# 1 - Right Propulsion
# 2 - Right Flight
# 3 - Left Ground Contact
# 4 - Left Propulsion
# 5 - Left Flight

def model_format(window_size, num_features):
    model = keras.Sequential([
        # Input
        keras.layers.Input(shape=(window_size, num_features)),

        keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        keras.layers.MaxPooling1D(pool_size=2),

        keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        keras.layers.MaxPooling1D(pool_size=2),

        keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),

        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(64, activation='relu'),

        # Output
        keras.layers.Dense(6, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_models(engines):
    window_size = 9

    for name in engines:
        if name == 'mediapipe' or name == 'mmpose':
            num_features = 50
            model = model_format(window_size, num_features)

        elif name == 'yolo26':
            num_features = 62
            model = model_format(window_size, num_features)

        model.summary()
        model.save(f'assets/phase_classifier_models/{name}_phase_classifier.keras')

def create_front_angle_model():
    window_size = 9

    from src.pose.yolo26_video_processor import get_data
    import numpy as np

    data, labels, _ = get_data(user_video='data/user_input/SHU-vf-normal-6.6mph.MOV')
    answer_key = np.load('assets/front_angle_labels/SHU-vf-normal-6.6mph.npy')
    model = model_format(window_size, 62)

    model.fit(data, answer_key, epochs=40, batch_size=64, validation_split=0.2, shuffle=True)
    model.save(f'assets/phase_classifier_models/mmpose_front_angle_phase_classifier.keras')

create_front_angle_model()