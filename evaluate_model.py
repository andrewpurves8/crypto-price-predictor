from tensorflow import keras
from split_data import *

def evaluate_model(model_name):
    model = keras.models.load_model(f'models/{model_name}')

    _, _, _, _, test_inputs, test_outputs = get_split_data('data/btc_usdt/1D/')

    total_squared_error = 0
    total_absolute_percentage_error = 0
    for i in range(len(test_inputs)):
        actual = test_outputs[i]
        prediction = model(np.array([test_inputs[i]])).numpy()[0][0]
        # print(f'Actual: {actual}')
        # print(f'Prediction: {prediction}')
        # print()

        error = actual - prediction
        total_squared_error += pow(error, 2)
        total_absolute_percentage_error += abs(error / actual * 100)

    mean_squared_error = total_squared_error / len(test_inputs)
    mean_absolute_percentage_error = total_absolute_percentage_error / len(test_inputs)

    # test_scores = model.evaluate(test_inputs, test_outputs, verbose=0)

    print('Model:', model_name)
    # print('Test loss:', test_scores[0])
    # print('Test accuracy:', test_scores[1])
    print('MSE:', mean_squared_error)
    print('MAPE:', mean_absolute_percentage_error)
    print()


if __name__ == '__main__':
    for model_name in range(3, 12):
        evaluate_model(model_name)