import seaborn as sns
import os
from matplotlib import pyplot as plt, image as mpimg
from logistic_regression.ImageLoader import ImageLoader
from logistic_regression.graphics import letter_map


def generate_class_heatmaps():
    loader = ImageLoader(shuffle=False).load_images()

    averages = {}
    for i, letter in letter_map.items():
        averages[letter] = loader.train_images.iloc[i * 3030:(i + 1) * 3030].mean()

    i = 1
    plt.figure(figsize=(10, 10))
    for letter, image in averages.items():
        plt.subplot(3, 3, i)
        ax = sns.heatmap(data=image.values.reshape((80, 80)))
        ax.set_title(letter)
        if i == 9:
            break
        i += 1
    plt.show()


def plot_class_images():
    i = 1
    plt.figure(figsize=(10, 10))
    for letter in os.scandir("../images/asl_alphabet_train/asl_alphabet_train"):
        for img in os.scandir(letter.path):
            ax = plt.subplot(3, 3, i)
            loaded_img = mpimg.imread(img.path)
            plt.imshow(loaded_img)
            ax.set_title(letter.name)
            if i == 9:
                break
            i += 1
            break
    plt.show()


def main():
    plot_class_images()
    generate_class_heatmaps()


if __name__ == "__main__":
    main()
