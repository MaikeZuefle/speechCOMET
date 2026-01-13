import os
import subset2evaluate.utils
import csv
import sklearn.model_selection

os.makedirs("data", exist_ok=True)

def fix_score(x):
    if x >= 1:
        x = x
    elif x < 0:
        x = max(0, 100 + 4*x)
    else:
        x = 100 * x
    x = max(min(x, 100), 0)
    return x

data_all = [
    {
        "src": x["src"],
        "mt": x["tgt"][sys],
        "score": fix_score(x["scores"][sys]["human"]),
        "src_audio": "",
    }
    for name, data in (subset2evaluate.utils.load_data_wmt_all(normalize=False) | subset2evaluate.utils.load_data_biomqm(normalize=False)).items()
    for x in data
    if "speech" not in x["domain"]
    for sys in x["tgt"]
]

# import matplotlib.pyplot as plt
# plt.hist([x["score"] for x in data_all])
# plt.show()
# exit()

data_train, data_dev = sklearn.model_selection.train_test_split(data_all, test_size=1000)

with open("data/text_train.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["score", "src", "mt", "src_audio"])
    writer.writeheader()
    writer.writerows(data_train)

with open("data/text_dev.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["score", "src", "mt", "src_audio"])
    writer.writeheader()
    writer.writerows(data_dev)