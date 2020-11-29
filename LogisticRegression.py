import pandas as pd
import pickle
import time
from ImageLoader import ImageLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def load_image_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    loader = ImageLoader()
    features, y_train = loader.load_images()
    return features, y_train


def scale_data(features: pd.DataFrame) -> pd.DataFrame:
    start = time.time()
    scaler = StandardScaler(copy=False)
    scaler.fit(features)
    x_train = scaler.transform(features)
    print("Scaler finished in:", time.time() - start, "seconds")
    return x_train


def grid_search_log_reg(x_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    start = time.time()
    grid = GridSearchCV(LogisticRegression(penalty="l1", solver="saga", tol=1e-2, max_iter=50),
                        param_grid={"C": [.5, .1, .05, .01]}, n_jobs=-1)
    grid.fit(x_train, y_train)
    print("Logistic regression grid search fit finished in:", (time.time() - start) / 3600, "hours")
    print("Best Params:", grid.best_params_, "Best Score:", grid.best_score_)
    return grid.best_estimator_


def save_theta(lr: LogisticRegression, filename: str = "log_reg_theta.pkl") -> None:
    with open(filename, "wb") as file:
        pickle.dump(lr, file)


def load_theta(filename: str = "log_reg_theta.pkl") -> LogisticRegression:
    with open(filename, "rb") as file:
        return pickle.load(file)


def train_lr_model() -> None:
    features, target = load_image_dataset()
    x_train = scale_data(features)
    lr = grid_search_log_reg(x_train, target)
    save_theta(lr)


if __name__ == "__main__":
    train_lr_model()
