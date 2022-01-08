from tensorflow import keras
from get_price_data import *

if __name__ == '__main__':
    model = keras.models.load_model('models/2_btc_usdt_1d_close_in_one_month_batch4_large')
    model.summary()
    # input = get_most_recent_price_data()
    # prediction = model(input).numpy()[0][0]
    # print(f"Predition: {prediction}")