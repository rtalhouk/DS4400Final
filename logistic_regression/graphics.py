import pandas as pd
import seaborn as sns
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from logistic_regression.ImageLoader import ImageLoader
from sklearn.preprocessing import label_binarize

letters = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
letters.insert(4, "del")
letters.insert(15, "nothing")
letters.insert(21, "space")

letter_map = {i: letter for i, letter in enumerate(letters)}


def lr_heatmap(lr):
    coefs = lr.coef_
    for cls in range(29):
        sns.heatmap(coefs[cls].reshape((80, 80)))
        plt.title(letters[cls])
        plt.show()


def score(lr, loader):
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


def main():
    with open("./log_reg_theta.pkl", "rb") as file:
        lr = pickle.load(file)

    loader = ImageLoader().load_images()
    # res = pd.read_csv("./lr_grid_search_results.csv")
    # sns.barplot(data=res, x="param_C", y="mean_test_score")
    # plt.show()
    # lr_heatmap()

    report, conf_matrix, rates, auc = score(lr, loader)
    graph_report(report, conf_matrix, rates, auc, prf_y_range=.5)


if __name__ == "__main__":
    main()
