import pandas as pd
import numpy as np
import pickle
import time
from ImageLoader import ImageLoader
from sklearn.svm import LinearSVC
from typing import Tuple
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

letters = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
letters.insert(4, "del")
letters.insert(15, "nothing")
letters.insert(21, "space")
letter_map = {i: letter for i, letter in enumerate(letters)}


def load_image_dataset(image_dir) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    loader = ImageLoader(image_dir)
    x_train, x_test, y_train, y_test = loader.load_images()
    x_train = loader.scale_data(x_train)
    x_test = loader.scale_data(x_test)
    return x_train, x_test, y_train, y_test


def fit_svm(x_train: pd.DataFrame, y_train: pd.Series) -> LinearSVC:
    start = time.time()
    svm = LinearSVC(loss="hinge", penalty="l2")
    svm.fit(x_train, y_train)
    print("LinearSVC fit finished in:", (time.time() - start) / 3600, "hours")
    return svm


def score_fit_svm(svm: LinearSVC, x_data: pd.DataFrame, y_data: pd.Series) -> Tuple[dict, np.array]:
    start = time.time()
    print(svm.score(x_data, y_data))
    predictions = svm.predict(x_data)
    report = classification_report(y_data, predictions, output_dict=True)
    conf_matrix = confusion_matrix(y_data, predictions, labels=letters)
    print(conf_matrix)
    print("SVC scoring finished in:", time.time() - start, "seconds")
    return report, conf_matrix



def save_theta(svm: LinearSVC, filename: str = "svm_theta_proba.pkl") -> None:
    with open(filename, "wb") as file:
        pickle.dump(svm, file)


def load_theta(filename: str = "svm_theta_no_reg.pkl") -> LinearSVC:
    with open(filename, "rb") as file:
        return pickle.load(file)


def train_svm_model(x_train, x_test, y_train, y_test) -> LinearSVC:
    svm = fit_svm(x_train, y_train)
    save_theta(svm)
    return model

def score_loaded():
    x_train, x_test, y_train, y_test = load_image_dataset("../images/asl_alphabet_train/asl_alphabet_train/")
    for file in ["svm_theta_no_reg.pkl", "svm_theta_0.1.pkl", "svm_theta_0.01.pkl", "svm_theta_0.001.pkl", "svm_theta.pkl"]:
        model = load_theta(file)
        print(model.C)
        print("\tTraining")
        train_report, train_conf = score_fit_svm(model, x_train, y_train)
        print("\tTesting")
        test_report, test_conf = score_fit_svm(model, x_test, y_test)


def score(model, x_test, y_test):
    pred = model.predict(x_test)
    class_probs = model.decision_function(x_test)
    report = classification_report(y_test,
                                   pred,
                                   output_dict=True)
    conf_matrix = confusion_matrix(y_test, pred)
    bin_targets = label_binarize(y_test, classes=letters)
    auc_score = roc_auc_score(bin_targets, class_probs)
    fpr, tpr, _ = roc_curve(bin_targets.ravel(), class_probs.ravel())
    return report, conf_matrix, (fpr, tpr), auc_score


def graph_report(report, conf_matrix, rates, auc, prf_y_range=.7):
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
    plt.figure(figsize=(30, 10))
    sns.heatmap(conf_matrix, annot=True, square=False,
                xticklabels=letter_map.values(), yticklabels=letter_map.values())
    plt.xticks(rotation=90)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.show()
    sns.lineplot(x=[0, 1], y=[0, 1])
    ax = sns.lineplot(x=rates[0], y=rates[1])
    ax.text(.75, 0, "AUC: {:.5f}".format(auc), bbox={"boxstyle": "round"})
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_image_dataset("../images/asl_alphabet_train/asl_alphabet_train/")
    model = load_theta()
    report, conf_matrix, rates, auc_score = score(model, x_test, y_test)
    graph_report(report, conf_matrix, rates, auc_score)
