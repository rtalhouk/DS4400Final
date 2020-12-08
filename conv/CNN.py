from keras import layers, Sequential
from keras.models import Sequential
from keras.losses import SparseCategoricalCrossentropy
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import time
import pathlib
from typing import Tuple, Any, Optional


def load_image_dataset(image_dir, batch_size, img_height, img_width) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    data_dir = pathlib.Path(image_dir)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    return train_ds, val_ds


def fit_cnn(train_ds: tf.data.Dataset, val_ds: tf.data.Dataset) -> Tuple[
    Sequential, Any]:
    start = time.time()
    data_augmentation = Sequential(
        [
         layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                      input_shape=(80, 
                                                                   80,
                                                                   3)),
         layers.experimental.preprocessing.RandomRotation(0.1),
         layers.experimental.preprocessing.RandomZoom(0.1),
         ])
    model = Sequential([
                        data_augmentation,
                        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(80, 80, 3)),
                        layers.Conv2D(80, (3, 3), activation='relu', input_shape=(80, 80, 1)),
                        layers.MaxPooling2D((2, 2)),
                        layers.Dropout(0.1),
                        layers.Conv2D(160, (3, 3), activation='relu'),
                        layers.MaxPooling2D((2, 2)),
                        layers.Conv2D(160, (3, 3), activation='relu'),
                        layers.Dropout(0.2),
                        layers.Flatten(),
                        layers.Dense(160, activation='relu'),
                        layers.Dense(29)
                        ])
    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_ds, epochs=10,
                        validation_data=val_ds)
    model.save("./model")
    print("CNN fit finished in:", (time.time() - start) / 3600, "hours")
    print("Model Summmary:", model.summary())
    return model, history


def evaluate_cnn(history) -> None:
    start = time.time()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(10)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    print("CNN scoring finished in:", time.time() - start, "seconds")


def save_theta(cnn: Sequential, filename: str = "cnn_theta.pkl") -> None:
    with open(filename, "wb") as file:
        pickle.dump(cnn, file)


def load_theta(filename: str = "cnn_theta.pkl") -> Sequential:
    with open(filename, "rb") as file:
        return pickle.load(file)


def train_cnn_model() -> None:
    train_ds, val_ds = load_image_dataset("/content/asl_alphabet/asl_alphabet_train/asl_alphabet_train/", 32, 80, 80)
    # x_test, y_test = load_image_dataset("./images/asl_alphabet_test/asl_alphabet_test/")
    # TODO: Get train/test data
    # x_test = scale_data(x_test)
    model, metrics = fit_cnn(train_ds, val_ds)
    evaluate_cnn(metrics)

if __name__ == "__main__":
    train_cnn_model()