import time
from logistic_regression.ImageLoader import ImageLoader
from keras import layers
from keras.models import Sequential
from keras.losses import SparseCategoricalCrossentropy
from typing import List
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay


def get_model(size_list: List[List[int]]) -> List[Sequential]:
    res = []
    for sizes in size_list:
        model = Sequential()
        model.add(layers.Input((80, 80, 1)))
        model.add(layers.Flatten())
        model.add(layers.experimental.preprocessing.Rescaling(1./255))
        for size in sizes:
            model.add(layers.Dropout(.5))
            model.add(layers.Dense(size, activation="relu"))
        model.add(layers.Dense(29, activation="softmax"))
        decay = ExponentialDecay(1e-3, decay_steps=10000, decay_rate=.9)
        model.compile(optimizer=Adam(learning_rate=decay),
                      loss=SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        res.append(model)

    return res


def generate_models():
    sizes = [
        [3200],
        [3200, 1600],
        [3200, 1600, 800]
    ]
    return get_model(sizes)


def main():
    accuracies = []
    models = generate_models()
    for model in models:
        print(model.summary())
    start = time.time()
    for model in models:
        train, valid = ImageLoader.load_images_for_keras()
        hist = model.fit(train, epochs=10, validation_data=valid)
        print(hist.history["accuracy"])
        train = model.evaluate(train)
        valid = model.evaluate(valid)
        accuracies.append((hist, train, valid))
    print("Finished in", (time.time() - start) / 3600, "hours")


if __name__ == "__main__":
    main()
