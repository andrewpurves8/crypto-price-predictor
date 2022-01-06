import numpy as np
import os
from random import random, seed

from tensorflow import keras
from tensorflow.keras import layers

seed(0)

train_split = 0.8
val_split = 0.1
test_split = 0.1


def read_data(files):
    inputs = []
    outputs = []
    for file_name in files:
        file_path = os.path.join(data_directory, file_name)

        if os.path.isfile(file_path):
            file = open(file_path, 'r')
            # There should be 505 lines in the file:
            # 500 correponding to the input - 100 days of klines
            # 145 unused - 29 days of klines
            # 5 correponding to the input - kline of final day
            lines = []
            for _ in range(130 * 5):
                lines.append(file.readline())

            input = []
            i = 0
            for line in lines[:-30 * 5]:
                input.append(float(line))
                i += 1
            inputs.append(input)

            output = []
            line = lines[-2]
            output.append(float(line))
            outputs.append(output)

    return np.array(inputs), np.array(outputs)


if __name__ == '__main__':
    data_directory = 'data/btc_usdt/1D/'
    files_train = []
    files_val = []
    files_test = []
    for file_name in os.listdir(data_directory):
        random_value = random()
        if (random_value < train_split):
            files_train.append(file_name)
        elif (random_value < train_split + val_split):
            files_val.append(file_name)
        else:
            files_test.append(file_name)

    train_inputs, train_outputs = read_data(files_train)
    val_inputs, val_outputs = read_data(files_val)
    test_inputs, test_outputs = read_data(files_test)

    model_inputs = keras.Input(shape=(500,))
    # initial
    # x = layers.Dense(256, activation='relu')(model_inputs)
    # x = layers.Dense(256, activation='relu')(x)
    # x = layers.Dense(128, activation='relu')(x)
    # x = layers.Dense(128, activation='relu')(x)
    # x = layers.Dense(64, activation='relu')(x)
    # small
    # x = layers.Dense(64, activation='relu')(model_inputs)
    # x = layers.Dense(64, activation='relu')(x)
    # large
    x = layers.Dense(256, activation='relu')(model_inputs)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    model_outputs = layers.Dense(1)(x)
    # model = keras.Model(inputs=model_inputs, outputs=model_outputs, name='3_btc_usdt_1d_close_in_one_month_batch4_small')
    model = keras.Model(inputs=model_inputs, outputs=model_outputs, name='3_btc_usdt_1d_close_in_one_month_batch4_large')
    model.summary()

    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.RMSprop(),
        metrics=['accuracy'],
    )

    history = model.fit(train_inputs, train_outputs, batch_size=4, epochs=10000, validation_data=(val_inputs, val_outputs))

    for i in range(len(test_inputs)):
        print(f'Actual: {test_outputs[i]}')
        print(
            f'Prediction: {model(np.array([test_inputs[i]])).numpy()[0][0]}'
        )

    test_scores = model.evaluate(test_inputs, test_outputs, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

    # model.save('models/3_btc_usdt_1d_close_in_one_month_batch4_small')
    model.save('models/3_btc_usdt_1d_close_in_one_month_batch4_large')