import fiona
from os import makedirs
from os.path import exists
import json


class Building:
    def __init__(self, shape_object):
        self.id = shape_object['id']
        geometry = shape_object['geometry']
        self.type = geometry['type']
        self.coordinates = geometry['coordinates']

        properties = shape_object['properties']
        self.bld_id = properties['bld_id']
        self.height = properties['height']
        self.region = properties['region']
        self.region_cat = properties['region_cat']

    def to_json(self):
        bld_json = {
            'id': self.id,
            'coords': self.coordinates,
        }

        return bld_json


class DataStorage:
    def __init__(self):
        self.MODIFIED_COLLECTION_PATH = "./data/collections/"
        if not exists(self.MODIFIED_COLLECTION_PATH):
            makedirs(self.MODIFIED_COLLECTION_PATH)

        self.collection = list(map(lambda item: Building(item),
            fiona.collection("data/yandex/bld_sample.shp")))

    def __getitem__(self, key):
        return self.collection[key]

    def apply_func(self, function):
        self.collection = list(map(function, self.collection))

    def save(self, name):
        with open(self.MODIFIED_COLLECTION_PATH + name, 'w') as file:
            json.dump(map(lambda bld: bld.to_json, self.collection), file)



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
