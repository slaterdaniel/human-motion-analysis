import numpy as np
from tensorflow import keras


def main():

    data = np.load('../../boeTreadpt1_capture.npy')
    answer_key = np.load('../../answer_key.npy')

    # Phases:
    # 0 - Right Ground Contact
    # 1 - Right Propulsion
    # 2 - Right Flight
    # 3 - Left Ground Contact
    # 4 - Left Propulsion
    # 5 - Left Flight

    model = keras.models.load_model('../../stride_model_50.keras')
    model.fit(data, answer_key, epochs=40, batch_size=64, validation_split=0.2, shuffle=True)

    output = model.predict(data)
    count = 0
    for out, ans in zip(np.argmax(output, axis=1), answer_key):
        if out != ans:
            count += 1
    print(f'{len(output) - count}/{len(output)}')

    model.save('stride_model_50.keras')

# main()
