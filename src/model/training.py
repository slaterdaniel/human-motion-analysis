from tensorflow import keras

def train_models(training_data, answer_key, engines):

    for name, data in zip(engines, training_data):
        print(len(data), len(answer_key))
        model = keras.models.load_model(f'../assets/{name}_phase_classifier.keras')
        model.fit(data, answer_key, epochs=40, batch_size=64, validation_split=0.2, shuffle=True)
        model.save(f'../assets/{name}_phase_classifier.keras')

