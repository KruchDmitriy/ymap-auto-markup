import fiona
from os import makedirs
from os.path import exists
import json


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


class MarkupStorage:
    def __init__(self):
        self.PATH_CHECKED_DATA = './data/checked/'
        if not exists(self.PATH_CHECKED_DATA):
            makedirs(self.PATH_CHECKED_DATA)

        self.markup = {}

    def append_building(self, user, building, is_bad):
        if type(building) is not Building:
            raise ValueError

        markup_json = building.to_json()
        markup_json['isBad'] = is_bad
        if user not in self.markup.keys():
            self.markup[user] = []

        self.markup[user].append(markup_json)

    def append_json(self, user, markup_json):
        if user not in self.markup.keys():
            self.markup[user] = []

        self.markup[user].append(markup_json)

    def dump(self, user, file_name='checked.json'):
        directory = self.PATH_CHECKED_DATA + user + '/'
        if not exists(directory):
            makedirs(directory)

        with open(directory + file_name, 'w') as markup_file:
            json.dump(self.markup[user], markup_file)

        del self.markup[user]
