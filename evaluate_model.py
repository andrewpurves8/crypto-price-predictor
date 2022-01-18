from tensorflow import keras
from split_data import *

def evaluate_model(ticker, model_name, period_model, period_data, exclude_volume, print_preditions=False):
    model = keras.models.load_model(f'models/{ticker}/{period_model}/{model_name}')

    _, _, _, _, test_inputs, test_outputs = get_split_data(f'data/{ticker}/{period_data}/', period_data, exclude_volume)

    total_absolute_percentage_error = 0
    for i in range(len(test_inputs)):
        actual = test_outputs[i]
        prediction = model(np.array([test_inputs[i]])).numpy()[0][0]
        if print_preditions:
            print(f'Actual: {actual}')
            print(f'Prediction: {prediction}')
            print()

        error = actual - prediction
        total_absolute_percentage_error += abs(error / actual * 100)

    mean_absolute_percentage_error = total_absolute_percentage_error / len(test_inputs)

    print()
    print('Model:', model_name)
    print('MAPE:', mean_absolute_percentage_error)


if __name__ == '__main__':
    evaluate_model('ETHUSDT', '1', period_model='1H', period_data='1H', exclude_volume=True, print_preditions=False)
