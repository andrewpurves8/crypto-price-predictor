import numpy as np
import os
from random import random, seed

train_split = 0.8
val_split = 0.1
test_split = 0.1

def read_data(files, data_directory, data_period, exclude_volume):
    inputs = []
    outputs = []
    total_entries = 0
    input_entries = 0
    if data_period == '1H':
        total_entries = 192
        input_entries = 144

    for file_name in files:
        file_path = os.path.join(data_directory, file_name)

        if os.path.isfile(file_path):
            file = open(file_path, 'r')
            lines = []
            for _ in range(total_entries * 5):
                lines.append(file.readline())

            input = []
            i = 0
            for line in lines[:input_entries * 5]:
                i += 1
                if exclude_volume and i % 5 == 0:
                    continue
                input.append(float(line))
            inputs.append(input)

            output = []
            line = lines[-2]
            output.append(float(line))
            outputs.append(output)

    return np.array(inputs), np.array(outputs)


def get_split_data(data_directory, data_period, exclude_volume):
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

    train_inputs, train_outputs = read_data(files_train, data_directory, data_period, exclude_volume)
    val_inputs, val_outputs = read_data(files_val, data_directory, data_period, exclude_volume)
    test_inputs, test_outputs = read_data(files_test, data_directory, data_period, exclude_volume)

    return (
        train_inputs,
        train_outputs,
        val_inputs,
        val_outputs,
        test_inputs,
        test_outputs
    )