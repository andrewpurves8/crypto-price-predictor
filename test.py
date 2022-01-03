import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(4,))
x = layers.Dense(64, activation="relu")(inputs)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs=inputs, outputs=outputs, name="test_model")
model.summary()

x_train = np.array(
    [
        [1, 2, 3, 4],
        [2, 5, 8, 3],
        [2, 5, 3, 6],
        [3, 7, 8, 3],
        [7, 6, 1, 5],
        [8, 4, 2, 3],
        [5, 7, 4, 1],
        [7, 5, 7, 3],
        [0, 3, 9, 6],
        [8, 2, 4, 7],
        [2, 6, 8, 4],
        [3, 8, 7, 3],
        [4, 9, 5, 6],
        [8, 7, 2, 4],
        [6, 5, 4, 2],
        [4, 4, 6, 6],
        [7, 3, 2, 8],
        [9, 2, 1, 9],
        [2, 5, 2, 8],
        [5, 7, 3, 9],
        [8, 9, 6, 2],
        [4, 6, 4, 4],
        [6, 8, 8, 9],
        [9, 4, 7, 4],
        [6, 2, 5, 5],
    ]
)

y_train = np.array(
    [
        30,
        48,
        45,
        53,
        42,
        34,
        35,
        50,
        57,
        52,
        54,
        52,
        61,
        44,
        36,
        54,
        51,
        52,
        50,
        64,
        52,
        44,
        82,
        54,
        45,
    ]
)

x_test = np.array(
    [[3, 3, 9, 7], [6, 5, 7, 2], [4, 4, 3, 4], [7, 8, 5, 7], [7, 9, 2, 4]]
)

y_test = np.array([64, 45, 37, 66, 47])

x_train = x_train.astype("float32")
y_train = y_train.astype("float32")
x_test = x_test.astype("float32")
y_test = y_test.astype("float32")

model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=1, epochs=100, validation_split=0.1)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

for i in range(len(x_test)):
    print("x_test[i]", x_test[i])
    print("y_test[i]", y_test[i])
    print(
        "model(np.array([x_test[i]])).numpy()",
        model(np.array([x_test[i]])).numpy()[0][0],
    )

model.save("path_to_my_model")
