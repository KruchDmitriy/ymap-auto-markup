import json
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
# import xgboost
from generate_task import Variator


def load_data():
    with open("data/statistics/bld_to_check.json", "r") as f:
        bld_to_check = json.load(f)

    X = []
    y = []

    for bld in bld_to_check:
        for task_id in bld_to_check[bld]:
            task = bld_to_check[bld][task_id]
            x = np.eye(3)

            if "transform" in task["meta"]:
                transforms = task["meta"]["transform"]
                if "point_shift" in transforms:
                    continue
                for trans in transforms:
                    if trans == "trans":  # translate
                        np.matmul(x, Variator.trans(transforms["trans"][0],
                                                    transforms["trans"][1]), x)
                    elif trans == "rotate":
                        np.matmul(x, Variator.rotate_matrix(transforms["rotate"]), x)
                    elif trans == "scale":
                        np.matmul(x, Variator.scale_matrix(transforms["scale"]), x)

            x = x.reshape(9)
            if task["meta"] == "original":
                # if we know that all original markup is good
                # X.append(x)
                # y.append(1)
                # continue

                contain_bad = False
                for markup in task["markup"]:
                    contain_bad = contain_bad or markup['isBad']
                if contain_bad:
                    continue

            for markup in task["markup"]:
                X.append(x)
                if markup["isBad"]:
                    y.append(0)
                else:
                    y.append(1)

    assert(len(X) == len(y))
    return X, y


def calc_perplexity(logit, X, y):
    pred = np.zeros(X.shape[0])
    proba = logit.predict_proba(X)

    pred[y == 0] = proba[y == 0][:, 0]
    pred[y == 1] = proba[y == 1][:, 1]

    log_pred = np.log(pred)
    mean = np.mean(log_pred)
    return math.exp(mean)


if __name__ == '__main__':
    X, y = load_data()
    X = np.array(X)
    y = np.array(y)
    y.reshape((1, y.shape[0]))

    logit = LogisticRegression()
    # logit = xgboost.XGBClassifier(max_depth=6, n_estimators=100)
    logit.fit(X, y)
    print(logit.coef_)
    print(calc_perplexity(logit, X, y))
