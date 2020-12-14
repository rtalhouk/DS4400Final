import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import random
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from logistic_regression.ImageLoader import ImageLoader
from sklearn.preprocessing import label_binarize
from typing import Tuple

# A list representing the letters
letters = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
letters.insert(4, "del")
letters.insert(15, "nothing")
letters.insert(21, "space")

# A map mapping the index to each letter
letter_map = {i: letter for i, letter in enumerate(letters)}


def score(lr, loader: ImageLoader) -> Tuple[np.array, np.array, Tuple[np.array, np.array], float]:
    """
    Score and get informational data used in the graph_report function.

    :param lr: The logistic regression to score
    :param loader: The imageloader to get image data from
    :return: A tuple containing the classification_report, confusion_matrix, a tuple of fpr and tpr values, and the auc
    """
    pred = lr.predict(loader.test_images)
    class_probs = lr.predict_proba(loader.test_images)
    report = classification_report(loader.test_classes,
                                   pred,
                                   output_dict=True)
    conf_matrix = confusion_matrix(loader.test_classes, pred)
    bin_targets = label_binarize(loader.test_classes, classes=letters)
    auc_score = roc_auc_score(bin_targets, class_probs)
    fpr, tpr, _ = roc_curve(bin_targets.ravel(), class_probs.ravel())
    return report, conf_matrix, (fpr, tpr), auc_score


def graph_report(report: np.array, conf_matrix: np.array, rates: Tuple[np.array, np.array], auc: float,
                 prf_y_range: float = .7) -> None:
    """
    Create graphs from the given information.

    :param report: A classification_report
    :param conf_matrix: A confusion_matrix
    :param rates: A tuple of FPR and TPR rates
    :param auc: The AUC
    :param prf_y_range: The range of Y values to show for the classification_report output
    :return:
    """
    # Create the precision, recall, and F1 graph
    data = pd.DataFrame(columns=["letter", "score", "type"])
    for letter in letters:
        precision = report[letter]["precision"]
        recall = report[letter]["recall"]
        f1 = report[letter]["f1-score"]
        data = data.append({"letter": letter,
                            "score": precision,
                            "type": "precision"}, ignore_index=True)
        data = data.append({"letter": letter,
                            "score": recall,
                            "type": "recall"}, ignore_index=True)
        data = data.append({"letter": letter,
                            "score": f1,
                            "type": "F1 score"}, ignore_index=True)
    sns.barplot(data=data, x="letter", y="score", hue="type")
    plt.xticks(rotation=90)
    plt.ylim(prf_y_range, 1)
    plt.show()

    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(30, 10))
    sns.heatmap(conf_matrix, annot=True, square=False,
                xticklabels=letter_map.values(), yticklabels=letter_map.values())
    plt.xticks(rotation=90)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.show()

    # Create a ROC graph with AUC
    sns.lineplot(x=[0, 1], y=[0, 1])
    ax = sns.lineplot(x=rates[0], y=rates[1])
    ax.text(.75, 0, "AUC: {:.5f}".format(auc), bbox={"boxstyle": "round"})
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


def plot_regularization_values(df) -> None:
    """
    Plots the regularization values and scores for a logistic regression.

    :return:
    """
    sns.lineplot(data=df, x=[1, .5, .1], y="mean_test_score")
    plt.show()


def plot_lr_heatmaps(lr):
    """
    Plots heatmaps for each class in a logistic regression's coefficient matrix.

    :param lr: The logistic regression
    :return:
    """
    i = 1
    plt.figure(figsize=(10, 10))
    for letter, image in random.sample(list(zip(lr.classes_, lr.coef_)), 9):
        plt.subplot(3, 3, i)
        ax = sns.heatmap(data=image.reshape((80, 80)))
        ax.set_title(letter)
        i += 1
    plt.show()


def main(lr, loader):
    report, conf_matrix, rates, auc = score(lr, loader)
    graph_report(report, conf_matrix, rates, auc, prf_y_range=.5)


if __name__ == "__main__":
    # plot_regularization_values()
    df_strict = pd.read_csv("./lr_grid_search_results_strict.csv")
    df_full = pd.read_csv("./lr_grid_search_results_full.csv")

    plot_regularization_values(df_strict)
    plot_regularization_values(df_full)

    with open("./log_reg_theta_strict.pkl", "rb") as file:
        lr_strict = pickle.load(file)
    with open("./log_reg_theta_full.pkl", "rb") as file:
        lr_full = pickle.load(file)
    loader = ImageLoader().load_images()

    plot_lr_heatmaps(lr_strict)
    main(lr_strict, loader)
    plot_lr_heatmaps(lr_full)
    main(lr_full, loader)
