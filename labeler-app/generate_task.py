from data_storage import DataStorage
from os import makedirs
from os.path import exists
import json


if __name__ == '__main__':
    dataStorage = DataStorage()
    directory = './data/markup_tasks/'
    if not exists(directory):
        makedirs(directory)

    for i in range(0, 200, 20):
        with open(directory + 'task' + str(i) + '.json', 'w') as f:
            f.write('[\n')

            sorted_storage = sorted(dataStorage.collection, key=lambda k: [k.coordinates[0][0][0], k.coordinates[0][0][1]])

            for i, bld in enumerate(sorted_storage[i: i + 20]):
                json.dump(bld.to_json(), f)
                if i != 19:
                    f.write(',\n')
            f.write('\n]\n')
