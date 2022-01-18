import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from custom_callback import *
from split_data import *
from evaluate_model import *

ticker = 'ETHUSDT'
model_name = '1'
data_period = '1H'
exclude_volume = True

if __name__ == '__main__':
    num_inputs = 0
    if data_period == '1H':
        num_inputs = 144

    values_per_entry = 4 if exclude_volume else 5


    (
        train_inputs,
        train_outputs,
        val_inputs,
        val_outputs,
        test_inputs,
        test_outputs
    ) = get_split_data(f'data/{ticker}/{data_period}/', data_period, exclude_volume)


    model_inputs = keras.Input(shape=(num_inputs * values_per_entry,))
    x = layers.Dense(4000, activation='relu', kernel_regularizer=keras.regularizers.L2(1e-5))(model_inputs)
    x = layers.Dense(4000, activation='relu', kernel_regularizer=keras.regularizers.L2(1e-5))(x)
    model_outputs = layers.Dense(1)(x)
    model = keras.Model(inputs=model_inputs, outputs=model_outputs, name=model_name)
    model.summary()

    model.compile(
        loss=keras.losses.MeanAbsolutePercentageError(),
        optimizer=keras.optimizers.RMSprop(),
        metrics=['mean_absolute_percentage_error'],
    )

    custom_callback = CustomCallback(model_name)
    history = model.fit(
        train_inputs,
        train_outputs,
        batch_size=64,
        epochs=500,
        validation_data=(val_inputs, val_outputs),
        callbacks=[custom_callback]
    )

    model.set_weights(custom_callback.best_weights)
    model.save(f'models/{ticker}/{data_period}/{model_name}')

    evaluate_model(ticker, model_name, data_period, data_period, exclude_volume)
    print("Best epoch:", custom_callback.best_epoch)

    epochs = np.array(custom_callback.epochs)
    losses = np.array(custom_callback.losses)
    val_losses = np.array(custom_callback.val_losses)

    plt.plot(epochs, losses)
    plt.plot(epochs, val_losses)
    plt.savefig(f'{model_name}.png')