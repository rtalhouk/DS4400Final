import json
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def generate_accuracy_graph(res, hist):
    sns.lineplot(data=res, x="model", y="acc", hue="type")
    plt.show()
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=hist[(hist["model"] == 0) & (hist["type"] == "acc")],
                 x="round", y="score", hue="subtype")
    plt.title("Training and Validation Accuracy")
    plt.subplot(1, 2, 2)
    sns.lineplot(data=hist[(hist["model"] == 0) & (hist["type"] == "loss")],
                 x="round", y="score", hue="subtype")
    plt.title("Training and Validation Loss")
    plt.show()


def main():
    res = pd.read_csv("./results.csv")
    history = pd.DataFrame(columns=["model", 'type', "subtype", 'score', 'round'])
    for i, hist in enumerate(res["history"]):
        data = json.loads(hist.replace("'", '"'))
        for j in range(len(data['loss'])):
            history = history.append({"model": i, 'type': 'loss', "subtype": "train_loss",
                                      'score': data['loss'][j],
                                      "round": j},
                                     ignore_index=True)
            history = history.append({"model": i, 'type': 'acc', "subtype": "train_acc",
                                      'score': data['accuracy'][j],
                                      "round": j},
                                     ignore_index=True)
            history = history.append({"model": i, 'type': "loss", "subtype": 'val_loss', 'score': data['val_loss'][j],
                                     "round": j},
                                     ignore_index=True)
            history = history.append({"model": i, 'type': "acc", "subtype": 'val_acc', 'score': data['val_accuracy'][j],
                                      "round": j},
                                     ignore_index=True)
    final_hist = pd.DataFrame(columns=["acc", "type"])
    for item in res.iterrows():
        final_hist = final_hist.append({"acc": item[1]["train_accuracy"],
                                        "type": "train"}, ignore_index=True)
        final_hist = final_hist.append({"acc": item[1]["val_accuracy"],
                                        "type": "val"}, ignore_index=True)
    final_hist["model"] = ["[3200]", "[3200]", "[3200, 1600]",
                           "[3200, 1600]", "[3200, 1600, 800]",
                           "[3200, 1600, 800]"]
    generate_accuracy_graph(final_hist, history)


if __name__ == "__main__":
    main()
