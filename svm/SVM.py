import pandas as pd
import numpy as np
import pickle
import time
from ImageLoader import ImageLoader
from sklearn.svm import LinearSVC
from typing import Tuple
from sklearn.metrics import classification_report, confusion_matrix

letters = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
letters.insert(4, "del")
letters.insert(15, "nothing")
letters.insert(21, "space")


def load_image_dataset(image_dir) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    loader = ImageLoader(image_dir)
    x_train, x_test, y_train, y_test = loader.load_images()
    x_train = loader.scale_data(x_train)
    x_test = loader.scale_data(x_test)
    return x_train, x_test, y_train, y_test


def fit_svm(x_train: pd.DataFrame, y_train: pd.Series) -> LinearSVC:
    start = time.time()
    svm = LinearSVC(loss="hinge", penalty="l2", C=0.0001)
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



def save_theta(svm: LinearSVC, filename: str = "svm_theta_0.0001.pkl") -> None:
    with open(filename, "wb") as file:
        pickle.dump(svm, file)


def load_theta(filename: str = "svm_theta.pkl") -> LinearSVC:
    with open(filename, "rb") as file:
        return pickle.load(file)


def train_svm_model() -> Tuple[dict, dict]:
    x_train, x_test, y_train, y_test = load_image_dataset("../images/asl_alphabet_train/asl_alphabet_train/")
    # TODO: Get train/test data
    svm = fit_svm(x_train, y_train)
    train_report, train_conf = score_fit_svm(svm, x_train, y_train)
    test_report, test_conf = score_fit_svm(svm, x_test, y_test)
    save_theta(svm)
    return train_report, test_report

def score_loaded():
    x_train, x_test, y_train, y_test = load_image_dataset("../images/asl_alphabet_train/asl_alphabet_train/")
    for file in ["svm_theta_no_reg.pkl", "svm_theta_0.1.pkl", "svm_theta_0.01.pkl", "svm_theta_0.001.pkl", "svm_theta.pkl"]:
        model = load_theta(file)
        print(model.C)
        print("\tTraining")
        train_report, train_conf = score_fit_svm(model, x_train, y_train)
        print("\tTesting")
        test_report, test_conf = score_fit_svm(model, x_test, y_test)


if __name__ == "__main__":
     score_loaded()
