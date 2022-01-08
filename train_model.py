import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from save_best_model import *
from split_data import *
from evaluate_model import *

model_name='12'

if __name__ == '__main__':
    (
        train_inputs,
        train_outputs,
        val_inputs,
        val_outputs,
        test_inputs,
        test_outputs
    ) = get_split_data('data/btc_usdt/1D/')

    model_inputs = keras.Input(shape=(500,))
    x = layers.Dense(2000, activation='relu')(model_inputs)
    # x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(2000, activation='relu')(x)
    # x = layers.Dropout(rate=0.2)(x)
    model_outputs = layers.Dense(1)(x)
    model = keras.Model(inputs=model_inputs, outputs=model_outputs, name=model_name)
    model.summary()

    model.compile(
        loss=keras.losses.MeanAbsolutePercentageError(),
        optimizer=keras.optimizers.RMSprop(),
        metrics=['mean_absolute_percentage_error'],
    )

    save_best_model = SaveBestModel(model_name)
    history = model.fit(
        train_inputs,
        train_outputs,
        batch_size=32,
        epochs=50000,
        validation_data=(val_inputs, val_outputs),
        callbacks=[save_best_model]
    )

    model.set_weights(save_best_model.best_weights)
    model.save(f'models/{model_name}')

    evaluate_model(model_name)