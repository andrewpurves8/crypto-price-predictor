from tensorflow import keras
from get_price_data import *

def infer(ticker, data_period, model_name):
    print()
    print(f'{ticker}/{model_name}')
    model = keras.models.load_model(f'models/{ticker}/{data_period}/{model_name}')
    input = get_most_recent_price_data(ticker, data_period, exclude_volume=True)
    prediction = model(input).numpy()[0][0]
    print(f'Predition: {prediction}')

if __name__ == '__main__':
    infer('BTCUSDT', '1H', '9')
    # infer('ETHUSDT', '1H', '1')