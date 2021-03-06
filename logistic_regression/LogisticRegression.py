import pandas as pd
import pickle
import time
from logistic_regression.ImageLoader import ImageLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def grid_search_log_reg(x_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Performs grid search on the logistic regression with  different values for c. Also saves
    the results of the grid search.

    :param x_train: The training dataset's images.
    :param y_train: The training dataset's classes.
    :return: The best logistic regression
    """
    start = time.time()
    grid = GridSearchCV(LogisticRegression(penalty="l1", solver="saga", tol=1e-2, max_iter=70),
                        param_grid={"C": [1, .5, .1]}, cv=3, n_jobs=3)
    grid.fit(x_train, y_train)
    print("Logistic regression grid search fit finished in:", (time.time() - start) / 3600, "hours")
    print("Best Params:", grid.best_params_, "Best Score:", grid.best_score_)
    print(grid.cv_results_)
    pd.DataFrame(data=grid.cv_results_).to_csv("lr_gs_res.csv")
    return grid.best_estimator_


def save_theta(lr: LogisticRegression, filename: str = "log_reg_theta.pkl") -> None:
    """
    Pickles the logistic regression.

    :param lr: The logistic regression to pickle.
    :param filename: The file to save the logistic regression to.
    :return:
    """
    with open(filename, "wb") as file:
        pickle.dump(lr, file)


def load_theta(filename: str = "log_reg_theta.pkl") -> LogisticRegression:
    """
    Loads the pickled logistic regression.

    :param filename: The name of the logistic regression file
    :return: The loaded logistic regression
    """
    with open(filename, "rb") as file:
        return pickle.load(file)


def train_lr_model() -> None:
    """
    Trains a logistic regression model using grid search.

    :return:
    """
    il = ImageLoader().load_images()
    lr = grid_search_log_reg(il.train_images, il.train_classes)
    save_theta(lr)
    print(lr.score(il.test_images, il.test_classes))


if __name__ == "__main__":
    train_lr_model()
