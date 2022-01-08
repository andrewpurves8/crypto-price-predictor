import numpy as np
import os
from random import random, seed

train_split = 0.8
val_split = 0.1
test_split = 0.1

def read_data(files, data_directory):
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


def get_split_data(data_directory):
    files_train = []
    files_val = []
    files_test = []
    seed(0)
    for file_name in os.listdir(data_directory):
        random_value = random()
        if (random_value < train_split):
            files_train.append(file_name)
        elif (random_value < train_split + val_split):
            files_val.append(file_name)
        else:
            files_test.append(file_name)

    train_inputs, train_outputs = read_data(files_train, data_directory)
    val_inputs, val_outputs = read_data(files_val, data_directory)
    test_inputs, test_outputs = read_data(files_test, data_directory)

    return (
        train_inputs,
        train_outputs,
        val_inputs,
        val_outputs,
        test_inputs,
        test_outputs
    )