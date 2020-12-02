from LogisticRegression import load_image_dataset, load_theta
from ImageLoader import ImageLoader
from PIL import Image


def main():
    il = ImageLoader()
    _, target = il.load_images()
    features = il.scale_data()
    # test = ImageLoader(image_dir="./images/temp_test").load_images()[0]
    # scaled_test = il.scale_data(test)
    lr = load_theta()
    lt = ImageLoader("./images/asl-alphabet-test")
    test_x, test_y = lt.load_images()
    test_x_scaled = il.scale_data(test_x)
    # print(lr.predict(scaled_test))
    print(lr.score(features, target))
    print(lr.score(test_x_scaled, test_y))


if __name__ == "__main__":
    main()
