import sys
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


if __name__ == '__main__':
    collection = list(map(lambda item: Building(item),
             fiona.collection(sys.argv[1])))
    directory = sys.argv[2]
    if not exists(directory):
        makedirs(directory)

    for i in range(0, 200, 20):
        with open(directory + '/task' + str(i) + '.json', 'w') as f:
            f.write('[\n')

            sorted_storage = sorted(collection, key=lambda k: [k.coordinates[0][0][0], k.coordinates[0][0][1]])

            for i, bld in enumerate(sorted_storage[i: i + 20]):
                json.dump(bld.to_json(), f)
                if i != 19:
                    f.write(',\n')
            f.write('\n]\n')
