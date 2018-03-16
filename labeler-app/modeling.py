import os.path
import json
import numpy as np
import math
import lightgbm as lgb
import pickle
import argparse
from affine import AffineMx
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score


def load_data(path_to_stat):
    with open(path_to_stat, "r") as f:
        bld_to_check = json.load(f)

    X = []
    y = []
    bld_id = []

    for bld in bld_to_check:
        for task_id in bld_to_check[bld]:
            task = bld_to_check[bld][task_id]
            x = np.zeros(4)
            point_shift_mean = 0

            if "transform" in task["meta"]:
                transforms = task["meta"]["transform"]
                for trans in transforms:
                    if trans == "trans":  # translate
                        x[0] = transforms["trans"][0]
                        x[1] = transforms["trans"][1]
                    elif trans == "rotate":
                        x[2] = transforms["rotate"]
                    elif trans == "scale":
                        x[3] = 1. - transforms["scale"]
                    elif trans == "point_shift":
                        sqrs = np.square(transforms["point_shift"])
                        point_shift_mean = np.mean(np.sqrt(sqrs[:, 0] + sqrs[:, 1]))

            x = np.abs(x)
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

            isBad = False
            for markup in task["markup"]:
                if markup["user"] == "solar":
                    continue
                isBad = isBad or markup["isBad"]

            X.append(x)
            bld_id.append(bld)
            if isBad:
                y.append(0)
            else:
                y.append(1)

    assert (len(X) == len(y))

    X = np.array(X)
    y = np.array(y)

    y.reshape((1, y.shape[0]))
    return X, y, bld_id


class AbstractModel:
    PATH_TO_STAT = "data/statistics/bld_to_check.json"

    def _fit(self, X, y):
        raise NotImplementedError()

    def fit(self, path_to_stat):
        if path_to_stat is None:
            path_to_stat = AbstractModel.PATH_TO_STAT
        X, y, _ = load_data(path_to_stat)
        self._fit(X, y)
        self.perplexity = calc_perplexity(self, X, y)

    def save(self, path=None):
        raise NotImplementedError()

    def load(self, path=None):
        raise NotImplementedError()

    def predict_proba(self, x):
        raise NotImplementedError()

    def predict_probas(self, X):
        raise NotImplementedError()

    def test(self, path_to_stat, log=True):
        if path_to_stat is None:
            path_to_stat = AbstractModel.PATH_TO_STAT

        X, y, bld_id = load_data(path_to_stat)
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, shuffle=True, random_state=42
            )

        self._fit(X_train, y_train)
        y_pred = self.predict_probas(X_test)
        auc = roc_auc_score(y_test, y_pred)

        class_pred = y_pred > 0.5
        accuracy = accuracy_score(y_test, class_pred)

        true_pos = np.logical_and(class_pred == 1, y_test == 1)
        false_pos = np.logical_and(class_pred == 1, y_test == 0)
        true_neg = np.logical_and(class_pred == 0, y_test == 0)
        false_neg = np.logical_and(class_pred == 0, y_test == 1)

        tps = []
        fps = []
        tns = []
        fns = []

        for i in range(class_pred.shape[0]):
            if true_pos[i]:
                tps.append(bld_id[i])
            if false_pos[i]:
                fps.append(bld_id[i])
            if true_neg[i]:
                tns.append(bld_id[i])
            if false_neg[i]:
                fns.append(bld_id[i])

        classification_meta = {
            'true_pos': tps,
            'false_pos': fps,
            'true_neg': tns,
            'false_neg': fns
        }

        if log:
            print('roc auc score ' + str(auc))
            print('accuracy ' + str(accuracy))
            print('confusion matrix\n' + str(np.array([[len(tps), len(fps)], [len(fns), len(tns)]])))

        return auc, accuracy, classification_meta


class TreesModel(AbstractModel):
    def __init__(self, model_params=None):
        self.DEFAULT_PATH_TO_MODEL = "data/trees.model"
        if model_params is None:
            self.model_params = {
                    'task': 'train',
                    'boosting_type': 'gbdt',
                    'max_depth': 3,
                    'min_data_in_leaf': 5,
                    'learning_rate': 0.01,
                    'verbose': -1,
                    'objective': 'binary'
                }
        self.num_rounds = 200

        self.perplexity = None

    def _fit(self, X, y):
        self.clf = lgb.train(self.model_params, lgb.Dataset(X, y), self.num_rounds)

    def save(self, path=None):
        if path is None:
            path = "data/trees.model"

        self.clf.save_model(path)

    def load(self, path=None):
        if path is None:
            path = self.DEFAULT_PATH_TO_MODEL

        if not os.path.isfile(path):
            raise FileNotFoundError("File with model not found, check path or run modeling.py --fit")

        self.clf = lgb.Booster(model_file=path)

    def predict_proba(self, x):
        return self.clf.predict(x.reshape((1, 5)))[0]

    def predict_probas(self, X):
        return self.clf.predict(X)


class LinearModel(AbstractModel):
    def __init__(self):
        self.DEFAULT_PATH_TO_MODEL = "data/linear.model"
        self.clf = LogisticRegression()
        self.perplexity = None
        self.weights = np.zeros(6)

    def _fit(self, X, y):
        self.clf.fit(X, y)
        self.weights = np.hstack((self.clf.coef_[0], self.clf.intercept_[0]))

    def save(self, path=None):
        if path is None:
            path = "data/linear.model"

        with open(path, "w") as f:
            for coef in self.clf.coef_[0]:
                f.write(str(coef) + ' ')
            f.write(str(self.clf.intercept_[0]))

    def load(self, path=None):
        if path is None:
            path = self.DEFAULT_PATH_TO_MODEL

        if not os.path.isfile(path):
            raise FileNotFoundError("File with model not found, check path or run modeling.py --fit")

        with open(path, 'r') as f:
            values = f.readlines()[0].split(' ')
            self.weights = np.array(list(map(float, values)))

    def predict_proba(self, x):
        summ = x[0] * self.weights[0] + \
                x[1] * self.weights[1] + \
                x[2] * self.weights[2] + \
                x[3] * self.weights[3] + \
                x[4] * self.weights[4] + \
                self.weights[5]

        return expit(summ)

    def predict_probas(self, X):
        probas = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            probas[i] = self.predict_proba(X[i])
        return probas


def calc_perplexity(model, X, y):
    proba = model.predict_probas(X)
    proba[y == 0] = 1. - proba[y == 0]
    log_proba = np.log(proba)
    mean = np.mean(log_proba)
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
    parser.add_argument('--model_type', required=True, help='"linear" or "trees"')
    parser.add_argument('--path_to_stat', help='path to statistics (file bld_to_check.json)')
    parser.add_argument('--path_out_model', help='path to save model')

    args = parser.parse_args()
    main(args)
