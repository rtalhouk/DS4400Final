import pandas as pd
import seaborn as sns
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
from logistic_regression.ImageLoader import ImageLoader

loader = ImageLoader().load_images()
letters = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
letters.insert(4, "del")
letters.insert(15, "nothing")
letters.insert(21, "space")


def lr_heatmap(lr):
    coefs = lr.coef_
    for cls in range(29):
        sns.heatmap(coefs[cls].reshape((80, 80)))
        plt.title(letters[cls])
        plt.show()


def score(lr, loader):
    report = classification_report(loader.test_classes,
                                   lr.predict(loader.test_images),
                                   output_dict=True)
    return report


def graph_report(report):
    data = pd.DataFrame(columns=["letter", "score", "type"])
    for letter in letters:
        precision = report[letter]["precision"]
        recall = report[letter]["recall"]
        f1 = 2 * (precision * recall) / (precision + recall)
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
    plt.show()


def main():
    with open("./log_reg_theta.pkl", "rb") as file:
        lr = pickle.load(file)

    # res = pd.read_csv("./lr_grid_search_results.csv")
    # sns.barplot(data=res, x="param_C", y="mean_test_score")
    # plt.show()
    # lr_heatmap()

    report = score(lr, loader)
    graph_report(report)

if __name__ == "__main__":
    main()
