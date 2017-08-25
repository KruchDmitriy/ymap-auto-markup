import numpy as np
import time

class ObliviousTree:
    def __init__(self, model, depth, coeff, log=False):
        self.features = np.zeros(shape=depth, dtype='int64')
        self.conditions = np.zeros(shape=depth, dtype='float64')
        self.values = np.zeros(shape=(2 ** depth), dtype='float64')
        self.coeff = coeff
        self.depth = depth

        for i in range(self.depth):
            line = model.readline()
            values = line.replace(',', '').split(' ')
            if values[0] == "feature:":
                self.features[i] = int(values[1])
                self.conditions[i] = float(values[5])

        line = model.readline()
        values = line.replace(',', '').split(' ')
        for value in values:
            idx, val, count = value.split(':')
            self.values[int(idx, 2)] = float(val)

        if log:
            print('features = ' + str(self.features))
            print('conditions = ' + str(self.conditions))
            print('values = ' + str(self.values))

    def predict(self, x):
        index = 0
        # while ()
            # index = int(x[self.features[i]] < self.conditions[i]) + index * 2
            # i += 1

        return self.values[index] * self.coeff


class Ensemble:
    def __init__(self, file, depth, log=True):
        self.log = log

        with open(file, 'r') as model:
            header = model.readline()
            num_trees = int(model.readline())

            print(num_trees)
            self.trees = []

            for i in range(num_trees):
                line = model.readline().split(' ')

                if line[0].find('ObliviousTree') != -1:
                    self.trees.append(ObliviousTree(model, depth, float(line[1])))

            print('model loaded successfully')

    def predict(self, x):
        start = time.time()

        result = 0
        for tree in self.trees:
            result += 1.0 #tree.predict(x)

        end = time.time()

        if self.log:
            print('elapsed time: ', end - start)

        return result

if __name__ == "__main__":
    ensemble = Ensemble('features21x21_300k.txt.model')

