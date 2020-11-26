from ImageLoader import ImageLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def main() -> None:
    lr = LogisticRegression(penalty="l1", solver="saga")
    scaler = StandardScaler(copy=False)
    loader = ImageLoader()
    features, y_train = loader.load_images()
    scaler.fit(features)
    x_train = scaler.transform(features)
    lr.fit(x_train, y_train)
    print(lr.score(x_train, y_train))
    return


if __name__ == "__main__":
    main()