import pandas as pd
import pickle
import time
from util.ImageLoader import ImageLoader
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def load_image_dataset(image_dir) -> Tuple[pd.DataFrame, pd.Series]:
    loader = ImageLoader(image_dir)
    features, y_train = loader.load_images()
    return features, y_train


def scale_data(features: pd.DataFrame) -> pd.DataFrame:
    start = time.time()
    scaler = StandardScaler(copy=False)
    scaler.fit(features)
    x_train = scaler.transform(features)
    print("Scaler finished in:", time.time() - start, "seconds")
    return x_train


def fit_svm(x_train: pd.DataFrame, y_train: pd.Series) -> LinearSVC:
    start = time.time()
    svm = LinearSVC(loss="hinge", penalty="l2", C=0.000001)
    svm.fit(x_train, y_train)
    print("LinearSVC fit finished in:", (time.time() - start) / 3600, "hours")
    return svm


def score_fit_svm(svm: LinearSVC, x_data: pd.DataFrame, y_data: pd.Series) -> None:
    start = time.time()
    print(svm.score(x_data, y_data))
    print("SVC scoring finished in:", time.time() - start, "seconds")


def save_theta(svm: LinearSVC, filename: str = "svm_theta.pkl") -> None:
    with open(filename, "wb") as file:
        pickle.dump(svm, file)


def load_theta(filename: str = "svm_theta.pkl") -> LinearSVC:
    with open(filename, "rb") as file:
        return pickle.load(file)


def train_svm_model() -> None:
    features, target = load_image_dataset("../images/asl_alphabet_train/asl_alphabet_train/")
    # TODO: Get train/test data
    x_train = scale_data(features)
    svm = fit_svm(x_train, target)
    score_fit_svm(svm, x_train, target)
    save_theta(svm)

def test_svm_model() -> None:
    features, target = load_image_dataset("./images/asl_alphabet_test/asl_alphabet_test/")
    x_test = scale_data(features)
    svm = load_theta("svm_theta.pkl")
    score_fit_svm(svm, x_test, target)


if __name__ == "__main__":
    train_svm_model()
    test_svm_model()