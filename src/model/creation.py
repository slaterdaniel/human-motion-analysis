from tensorflow import keras

# Phases:
# 0 - Right Ground Contact
# 1 - Right Propulsion
# 2 - Right Flight
# 3 - Left Ground Contact
# 4 - Left Propulsion
# 5 - Left Flight

def create_model():
    window_size = 9
    num_features = 50

    model = keras.Sequential([
        # Layer1
        keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(window_size, num_features)),
        keras.layers.MaxPooling1D(pool_size=2),

        # Layer2
        keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        keras.layers.MaxPooling1D(pool_size=2),

        # Layer3
        keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),

        # Output
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(6, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.save('../assets/phase_classifier50.keras')
