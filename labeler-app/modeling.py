import json
import numpy as np
import math
import xgboost
import pickle
import argparse
from affine import AffineMx
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score


def load_data(path_to_stat):
    with open(path_to_stat, "r") as f:
        bld_to_check = json.load(f)

    X = []
    y = []

    for bld in bld_to_check:
        for task_id in bld_to_check[bld]:
            task = bld_to_check[bld][task_id]
            # x = np.eye(3)
            x = np.zeros(4)
            point_shift_mean = 0

            if "transform" in task["meta"]:
                transforms = task["meta"]["transform"]
                for trans in transforms:
                    if trans == "trans":  # translate
                        # x = np.matmul(x, AffineMx.trans(transforms["trans"][0],
                        #                                 transforms["trans"][1]))
                        x[0] = transforms["trans"][0]
                        x[1] = transforms["trans"][1]
                    elif trans == "rotate":
                        # x = np.matmul(x, AffineMx.rotate(transforms["rotate"]))
                        x[2] = transforms["rotate"]
                    elif trans == "scale":
                        # x = np.matmul(x, AffineMx.scale(transforms["scale"]))
                        x[3] = 1. - transforms["scale"]
                    elif trans == "point_shift":
                        sqrs = np.square(transforms["point_shift"])
                        point_shift_mean = np.mean(np.sqrt(sqrs[:, 0] + sqrs[:, 1]))

            # x = x.reshape(9)
            x = np.abs(x)
            x = np.hstack((x, point_shift_mean))
            # print(x)

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
    PATH_TO_STAT = "data/statistics/bld_to_check.json"

    def fit(self, path_to_stat):
        if path_to_stat is None:
            path_to_stat = AbstractModel.PATH_TO_STAT
        X, y = load_data(path_to_stat)
        self.clf.fit(X, y)
        self.perplexity = calc_perplexity(self.clf, X, y)

    def save(self, path):
        if path is None:
            path = self.DEFAULT_PATH_TO_MODEL
        pickle.dump(self.clf, open(path, "wb"))

    def load(self, path):
        if path is None:
            path = self.DEFAULT_PATH_TO_MODEL
        try:
            self.clf = pickle.load(open(path, "rb"))
        except FileNotFoundError:
            print("File with xgb model not found, please run modeling.py --fit")

    def predict_probas(self, X):
        return self.clf.predict_proba(X)

    def test(self, path_to_stat):
        if path_to_stat is None:
            path_to_stat = AbstractModel.PATH_TO_STAT

        X, y = load_data(path_to_stat)
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, shuffle=False
            )

        self.clf.fit(X_train, y_train)
        y_pred = self.predict_probas(X_test)
        auc = roc_auc_score(y_test, y_pred[:,1])
        print('roc auc score ' + str(auc))

        class_pred = y_pred[:,1] > 0.4
        accuracy = accuracy_score(y_test, class_pred)
        print('accuracy ' + str(accuracy))



class TreesModel(AbstractModel):
    def __init__(self, model_params=None):
        self.DEFAULT_PATH_TO_MODEL = "data/xgb.model"
        if model_params is None:
            model_params = {"max_depth": 6, "n_estimators": 50}
        self.clf = xgboost.XGBClassifier(max_depth=model_params["max_depth"],
                                              n_estimators=model_params["n_estimators"])
        self.perplexity = None


class LinearModel(AbstractModel):
    def __init__(self):
        self.DEFAULT_PATH_TO_MODEL = "data/linear.model"
        self.clf = LogisticRegression()
        self.perplexity = None

    def save(self, path):
        super().save(path)
        if path is None:
            path = "data/linear.params"

        with open(path, "w") as f:
            for coef in self.clf.coef_[0]:
                f.write(str(coef) + ' ')
            f.write(str(self.clf.intercept_[0]))


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
        if args.model_type == "linear":
            model = LinearModel()
        elif args.model_type == "trees":
            model = TreesModel()
        model.fit(args.path_to_stat)
        model.save(args.path_out_model)

        print("Model perplexity " + str(model.perplexity))

        if args.model_type == "linear":
            print('coeffs' + str(model.clf.coef_) + ' ' + str(model.clf.intercept_))
    elif args.test:
        if args.model_type == "linear":
            model = LinearModel()
        elif args.model_type == "trees":
            model = TreesModel()

        model.test(args.path_to_stat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', action='store_true', help='fit model on user statistics')
    parser.add_argument('--test', action='store_true', help='calculate model accuracy'
                                                            '(and AUC) on splitted set')
    parser.add_argument('--model_type', help='"linear" or "trees"')
    parser.add_argument('--path_to_stat', help='path to statistics (file bld_to_check.json)')
    parser.add_argument('--path_out_model', help='path to save model')

    args = parser.parse_args()
    main(args)
