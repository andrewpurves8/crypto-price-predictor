from datetime import datetime
import glob
import os
import time
import numpy as np
from dotenv import load_dotenv
from binance import Client

ms_per_hour = 60 * 60 * 1000
ms_per_day = ms_per_hour * 24

def create_binance_client():
    load_dotenv()

    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

    return Client(BINANCE_API_KEY, BINANCE_API_SECRET)


def get_most_recent_price_data(ticker, data_period, exclude_volume):
    client = create_binance_client()

    if data_period == '1H':
        # 168 hours (7 days) of input
        num_inputs = 168
        interval = ms_per_hour
        interval_constant = Client.KLINE_INTERVAL_1HOUR

    # Read most recent data up until yesterday
    timestamp_end = round(time.time() * 1000)
    timestamp_start = timestamp_end - num_inputs * interval

    klines = client.get_historical_klines(
        ticker,
        interval_constant,
        timestamp_start,
        timestamp_end
    )

    data = []
    for kline in klines:
        data.append(float(kline[1])) # open
        data.append(float(kline[2])) # high
        data.append(float(kline[3])) # low
        data.append(float(kline[4])) # close
        if not exclude_volume:
            data.append(float(kline[5])) # volume

    return np.array([data])


def save_historical_price_data(ticker, data_period):
    client = create_binance_client()

    timestamp_listing = 0
    timestamp_final = 1641247200001

    file_list = glob.glob(f'data/{ticker}/{data_period}/*')
    for file in file_list:
        os.remove(file)

    if data_period == '1H':
        # 168 hours (7 days) input, 23 hours skipped, 1 hour output
        num_entries = 192
        interval_constant = Client.KLINE_INTERVAL_1HOUR

    klines = client.get_historical_klines(
        ticker,
        interval_constant,
        timestamp_listing,
        timestamp_final
    )

    for i in range(len(klines) - num_entries):
        klines_slice = klines[i : i + num_entries]

        timestamp_actual = klines_slice[0][0]
        date_time = datetime.fromtimestamp(timestamp_actual / 1000)
        print(date_time.strftime("%d/%m/%Y, %H:%M:%S"))

        with open(f'data/{ticker}/{data_period}/' + str(timestamp_actual) + '.txt', 'w') as f:
            for kline in klines_slice:
                f.write(kline[1] + '\n') # open
                f.write(kline[2] + '\n') # high
                f.write(kline[3] + '\n') # low
                f.write(kline[4] + '\n') # close
                f.write(kline[5] + '\n') # volume

        progress = (i / (len(klines) - num_entries)) * 100.0
        print(f'{progress}%')
        print()


if __name__ == '__main__':
    save_historical_price_data('ETHUSDT', '1H')