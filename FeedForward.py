from ImageLoader import ImageLoader
from keras import layers
from keras.models import Sequential
from keras.losses import MeanSquaredError


def main():
    model = Sequential()
    model.add(layers.Dense(80*20, activation="relu", input_shape=(80*80,)))
    model.add(layers.Dense(80*8, activation="relu"))
    model.add(layers.Dense(80, activation="relu"))
    model.add(layers.Dense(29, activation="relu"))
    print(model.summary())
    model.compile(optimizer="adam", loss=MeanSquaredError(), metrics=["accuracy"])
    il = ImageLoader()
    features, targets = il.load_images()
    hist = model.fit(x=features.values / 255.0, y=targets.values, epochs=10, validation_split=.2)
    print(hist.history["accuracy"])
    print(model.evaluate(features.values / 255.0, targets.values))
    il2 = ImageLoader(image_dir="./images/asl-alphabet-test")
    train_x, train_y = il2.load_images()
    print(model.evaluate(train_x.values / 255.0, train_y.values))


if __name__ == "__main__":
    main()
