import numpy as np
from tensorflow import keras

BOE1 = 'boeTreadpt1_capture.npy'
data = np.load(BOE1)
answer_key = np.load('../../answer_key.npy')

# Phases:
# 0 - Right Ground Contact
# 1 - Right Propulsion
# 2 - Right Flight
# 3 - Left Ground Contact
# 4 - Left Propulsion
# 5 - Left Flight

print(data.shape)
print(answer_key.shape)

window_size = len(data[0])
num_features = len(data[0,0])

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

output = model.predict(data)
count = 0
for out, ans in zip(np.argmax(output, axis=1), answer_key):
    if out != ans:
        count += 1
print(f'{len(output) - count}/{len(output)}')

model.save('stride_model_50.keras')
