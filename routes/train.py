# by M.A.G.N.E.U.M
# This code builds a facial emotion recognition model using tensorflow keras and fer2013.csv

import os
import logging
import subprocess
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from colorama import Fore, Style
from keras_tuner import RandomSearch

logger = logging.getLogger("tensorflow.python.framework")
logger.setLevel(logging.CRITICAL)


def main():
    def clear_screen():
        if os.name == "nt":
            subprocess.call(["cls"])
        else:
            subprocess.call(["clear"])

    clear_screen()
    batch_size = int(
        input(Fore.YELLOW + Style.BRIGHT + "Enter the batch size: " + Style.RESET_ALL)
    )
    max_trials = int(
        input(Fore.YELLOW + Style.BRIGHT + "Enter the max trails: " + Style.RESET_ALL)
    )
    max_epochs = int(
        input(Fore.YELLOW + Style.BRIGHT + "Enter the max epochs: " + Style.RESET_ALL)
    )
    max_patience = int(
        input(Fore.YELLOW + Style.BRIGHT + "Enter the max patience: " + Style.RESET_ALL)
    )
    main_dir = input(
        Fore.YELLOW + Style.BRIGHT + "Enter the directory name: " + Style.RESET_ALL
    )

    if not os.path.exists(main_dir):
        os.mkdir(main_dir)
        print(
            Fore.GREEN
            + Style.BRIGHT
            + f"The directory '{main_dir}' has been created."
            + Style.RESET_ALL
        )
    else:
        print(
            Fore.RED
            + Style.BRIGHT
            + f"The directory '{main_dir}' already exists."
            + Style.RESET_ALL
        )
    if not os.path.exists(main_dir + "/view"):
        os.mkdir(main_dir + "/view")
        print(Fore.GREEN + "Created view folder." + Style.RESET_ALL)
    if not os.path.exists(main_dir + "/bin"):
        os.mkdir(main_dir + "/bin")
        print(Fore.GREEN + "Created bin folder." + Style.RESET_ALL)
    if not os.path.exists(main_dir + "/model"):
        os.mkdir(main_dir + "/model")
        print(Fore.GREEN + "Created model folder." + Style.RESET_ALL)
    print("All folders have been created.")

    data = pd.read_csv("database/fer2013.csv")
    clear_screen()
    print(Fore.YELLOW + "Data:" + Style.RESET_ALL)
    print(data)

    def preprocess_data(data):
        X_Trainer = data.pixels.apply(
            lambda x: np.fromstring(x, sep=" ", dtype=np.float32)
        ).values
        X_Trainer = np.stack(X_Trainer).reshape(-1, 48, 48, 1)
        Y_Trainer = data.emotion.values
        Y_Trainer = keras.utils.to_categorical(Y_Trainer, num_classes=7)
        return X_Trainer, Y_Trainer

    X_Trainer, Y_Trainer = preprocess_data(data)
    print(Fore.YELLOW + "X_Trainer shape:" + Style.RESET_ALL, X_Trainer.shape)
    print(Fore.YELLOW + "Y_Trainer shape:" + Style.RESET_ALL, Y_Trainer.shape)

    def Hyper_Builder(hp):
        model = keras.models.Sequential()
        model.add(
            keras.layers.Conv2D(
                hp.Int("conv_filters_1", min_value=16, max_value=128, step=16),
                (3, 3),
                padding="same",
                activation="relu",
            )
        )
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.25))
        for i in range(hp.Int("num_layers", min_value=2, max_value=6)):
            model.add(
                keras.layers.Conv2D(
                    hp.Int(f"conv_filters_{i}", min_value=16, max_value=128, step=16),
                    (3, 3),
                    padding="same",
                    activation="relu",
                )
            )
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Flatten())
        model.add(
            keras.layers.Dense(
                hp.Int("dense_units", min_value=64, max_value=512, step=64),
                activation="relu",
            )
        )
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(7, activation="softmax"))
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(model)
        return model

    all_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=max_patience, restore_best_weights=True
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=max_patience, restore_best_weights=True
        ),
    ]

    Hyper_Tuner = RandomSearch(
        Hyper_Builder,
        directory=main_dir,
        project_name="bin",
        max_trials=max_trials,
        executions_per_trial=3,
        objective="val_accuracy",
    )
    Hyper_Tuner.search(
        X_Trainer,
        Y_Trainer,
        epochs=max_epochs,
        validation_split=0.2,
        batch_size=batch_size,
        callbacks=all_callbacks,
    )

    clear_screen()
    Best_Model = Hyper_Tuner.get_best_models(num_models=1)[0]
    Best_Model.fit(
        X_Trainer,
        Y_Trainer,
        epochs=max_epochs,
        validation_split=0.2,
        batch_size=batch_size,
        callbacks=all_callbacks,
    )
    print(Best_Model.summary())

    model_json = Best_Model.to_json()
    with open(main_dir + "/model/Fer_model.json", "w") as json_file:
        json_file.write(model_json)
    Best_Model.save_weights(main_dir + "/model/Fer_model.h5")


if __name__ == "__main__":
    main()
