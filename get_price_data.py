import glob
import os
import time
import numpy as np
from dotenv import load_dotenv
from binance import Client

ms_per_day = 24 * 60 * 60 * 1000

def create_binance_client():
    load_dotenv()

    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

    return Client(BINANCE_API_KEY, BINANCE_API_SECRET)


def get_most_recent_price_data():
    client = create_binance_client()

    # Read most recent data up until yesterday
    timestamp_end = round(time.time() * 1000) - ms_per_day
    timestamp_start = timestamp_end - 100 * ms_per_day

    klines = client.get_historical_klines(
        'BTCUSDT',
        Client.KLINE_INTERVAL_1DAY,
        timestamp_start,
        timestamp_end
    )

    data = []
    for kline in klines:
        data.append(float(kline[1])) # open
        data.append(float(kline[2])) # high
        data.append(float(kline[3])) # low
        data.append(float(kline[4])) # close
        data.append(float(kline[5])) # volume

    return np.array([data])


def save_historical_price_data():
    client = create_binance_client()

    # Millisecond timestamp at start of BTCUSDT history, 18 August 2017
    timestamp_listing = 1503007200000
    # Millisecond timestamp at end of sampling period, 4 Jan 2022
    timestamp_final = 1641247200001

    file_list = glob.glob('data/btc_usdt/1D/*')
    for file in file_list:
        os.remove(file)

    # step = 1 day
    for timestamp_start in range(timestamp_listing, timestamp_final, ms_per_day):
        # +130 days (100 days input, 29 days unused, 1 day as output)
        timestamp_end = timestamp_start + 130 * ms_per_day

        klines = client.get_historical_klines(
            'BTCUSDT',
            Client.KLINE_INTERVAL_1DAY,
            timestamp_start,
            timestamp_end
        )

        if (len(klines) < 130):
            # Less than the desired 130 days means we've run out of full-length samples
            break

        timestamp_actual = klines[0][0]

        with open('data/btc_usdt/1D/' + str(timestamp_actual) + '.txt', 'w') as f:
            for kline in klines:
                f.write(kline[1] + '\n') # open
                f.write(kline[2] + '\n') # high
                f.write(kline[3] + '\n') # low
                f.write(kline[4] + '\n') # close
                f.write(kline[5] + '\n') # volume


if __name__ == '__main__':
    save_historical_price_data()