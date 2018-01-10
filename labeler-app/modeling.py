import json
import numpy as np
import math
import xgboost
import pickle
import argparse
from affine import AffineMx
from sklearn.linear_model import LogisticRegression


def load_data(path_to_stat):
    with open(path_to_stat, "r") as f:
        bld_to_check = json.load(f)

    X = []
    y = []

    for bld in bld_to_check:
        for task_id in bld_to_check[bld]:
            task = bld_to_check[bld][task_id]
            x = np.eye(3)
            point_shift_mean = 0

            if "transform" in task["meta"]:
                transforms = task["meta"]["transform"]
                for trans in transforms:
                    if trans == "trans":  # translate
                        x = np.matmul(x, AffineMx.trans(transforms["trans"][0],
                                                        transforms["trans"][1]))
                    elif trans == "rotate":
                        x = np.matmul(x, AffineMx.rotate(transforms["rotate"]))
                    elif trans == "scale":
                        x = np.matmul(x, AffineMx.scale(transforms["scale"]))
                    elif trans == "point_shift":
                        sqrs = np.square(transforms["point_shift"])
                        point_shift_mean = np.mean(np.sqrt(sqrs[:, 0] + sqrs[:, 1]))

            x = x.reshape(9)
            x = np.hstack((x, point_shift_mean))

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
                if markup["user"] == "solar":
                    continue

                X.append(x)
                if markup["isBad"]:
                    y.append(0)
                else:
                    y.append(1)

    assert (len(X) == len(y))
    X = np.array(X)
    y = np.array(y)
    y.reshape((1, y.shape[0]))
    return X, y


class AbstractModel:
    def fit(self, path_to_stat):
        if path_to_stat is None:
            path_to_stat = "data/statistics/bld_to_check.json"
        X, y = load_data(path_to_stat)
        self.clf.fit(X, y)
        self.perplexity = calc_perplexity(self.clf, X, y)

    def save(self, path):
        if path is None:
            path = AbstractModel.DEFAULT_PATH_TO_MODEL
        pickle.dump(self.clf, open(path, "wb"))

    def load(self, path):
        if path is None:
            path = AbstractModel.DEFAULT_PATH_TO_MODEL
        try:
            self.clf = pickle.load(open(path, "rb"))
        except FileNotFoundError:
            print("File with xgb model not found, please run modeling.py --fit")

    def predict_probas(self, X):
        return self.clf.predict_proba(X)

class TreesModel(AbstractModel):
    DEFAULT_PATH_TO_MODEL = "data/xgb.model"

    def __init__(self, model_params=None):
        if model_params is None:
            model_params = {"max_depth": 6, "n_estimators": 100}
        self.clf = xgboost.XGBClassifier(max_depth=model_params["max_depth"],
                                              n_estimators=model_params["n_estimators"])
        self.perplexity = None

class LinearModel(AbstractModel):
    DEFAULT_PATH_TO_MODEL = "data/linear.model"

    def __init__(self):
        self.clf = LogisticRegression()
        self.perplexity = None

def calc_perplexity(model, X, y):
    pred = np.zeros(X.shape[0])
    proba = model.predict_proba(X)

    pred[y == 0] = proba[y == 0][:, 0]
    pred[y == 1] = proba[y == 1][:, 1]

    log_pred = np.log(pred)
    mean = np.mean(log_pred)
    return math.exp(mean)


def main(args):
    if args.fit:
        model = TreesModel()
        model.fit(args.path_to_stat)
        model.save(args.path_out_model)
        print("Model perplexity " + str(model.perplexity))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', action='store_true', help='fit xgb model on user statistics')
    parser.add_argument('--path_to_stat', help='path to statistics (file bld_to_check.json)')
    parser.add_argument('--path_out_model', help='path to save model')

    args = parser.parse_args()
    main(args)
