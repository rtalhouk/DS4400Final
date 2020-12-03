import time
from ImageLoader import ImageLoader
from keras import layers
from keras.models import Sequential
from keras.losses import MeanSquaredError
from typing import List
from keras.optimizers import Adam


def get_model(size_list: List[List[int]]) -> List[Sequential]:
    res = []
    for sizes in size_list:
        model = Sequential()
        model.add(layers.Input((200, 200, 1)))
        model.add(layers.Flatten())
        model.add(layers.experimental.preprocessing.Rescaling(1./255))
        for size in sizes:
            model.add(layers.Dense(size, activation="relu"))
        model.add(layers.Dense(29, activation="softmax"))
        model.compile(optimizer=Adam(learning_rate=.01), loss=MeanSquaredError(),
                      metrics=['accuracy'])
        res.append(model)

    return res


def generate_models():
    sizes = [
             [80 * 80, 80 * 60, 80 * 40, 80 * 20, 80 * 10, 80 * 5, 80 * 2, 80,
              60, 50, 45, 40, 35, 30],
             [80 * 80, 80 * 60, 80 * 40, 80 * 20, 80 * 5, 80, 60, 40, 35],
             [80 * 80, 80 * 40, 80 * 20, 80 * 10, 80 * 5, 80, 60, 35],
             [80 * 40, 80 * 10, 80 * 5, 80, 55],
             [80 * 40, 80 * 10, 80, 40]
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
        accuracies.append((train, valid))
    print("Finished in", (time.time() - start) / 3600, "hours")


if __name__ == "__main__":
    main()
