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
    _, targets = il.load_images()
    features = il.scale_data()
    hist = model.fit(x=_.values / 255.0, y=targets.values, epochs=10, validation_split=.2)
    print(model.evaluate())
    print(hist.history["accuracy"])


if __name__ == "__main__":
    main()
