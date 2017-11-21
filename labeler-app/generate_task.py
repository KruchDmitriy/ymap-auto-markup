import fiona
import json
import argparse
import numpy as np
from os import makedirs
from os.path import exists
from scipy.spatial import KDTree
from numpy.random import randint
from shapely.geometry import Polygon


class Collection:
    EPSILON = 1e-7

    def __init__(self, path_to_data):
        collection = list(map(lambda item: Building(item),
                              fiona.collection(path_to_data)))
        self.sorted_storage = sorted(collection, key=lambda k:
                                     [k.raw_coords[0][0], k.raw_coords[0][1]])

        self._create_centers()
        self.length = len(self.sorted_storage)
        self.kd_tree = KDTree(self.centers)

    def __getitem__(self, item):
        return self.sorted_storage[item]

    def _create_centers(self):
        centers_set = set()
        self.centers = np.zeros(shape=(len(self.sorted_storage), 2))

        for i, building in enumerate(self.sorted_storage):
            center = building.center

            while center in centers_set:
                center = (center[0] + Collection.EPSILON, center[1])

            centers_set.add(center)
            self.centers[i][0] = center[0]
            self.centers[i][1] = center[1]

    def get_nearest(self, point, n_nearest):
        _, center_idxs = self.kd_tree.query(point, n_nearest)
        return [self.sorted_storage[idx] for idx in center_idxs]


class Building:
    def __init__(self, shape_object):
        self.id = shape_object['id']
        geometry = shape_object['geometry']
        self.type = geometry['type']
        self.polygon = Polygon(geometry['coordinates'][0])
        self.raw_coords = geometry['coordinates'][0]

        properties = shape_object['properties']
        self.bld_id = properties['bld_id']
        self.height = properties['height']
        self.region = properties['region']
        self.region_cat = properties['region_cat']

    @property
    def center(self):
        center_point = self.polygon.centroid.coords
        return center_point[0][0], center_point.xy[1][0]

    def flush(self, file_desc):
        bld_json = {
            'id': self.id,
            'coords': self.raw_coords,
        }

        json.dump(bld_json, file_desc)


class TaskGenerator:
    SAMPLING_FACTOR = 5
    TRYING_FACTOR = 10

    def __init__(self, collection, output_dir, task_size):
        self.collection = collection
        self.output_dir = output_dir
        self.tasks = []
        self.involved_polygons = []
        self.task_size = task_size

    def generate(self, num_mixed, num_sequential):
        tasks = []
        for i in range(num_mixed):
            tasks.append(self._generate_mixed_task())

        for i in range(num_sequential):
            tasks.append(self._generate_sequential_task())

        return tasks

    def _generate_mixed_task(self):
        anchor_idx = randint(0, self.collection.length - 1)
        anchor = self.collection[anchor_idx].center

        nearest_buildings = self.collection.get_nearest(anchor, self.task_size * TaskGenerator.SAMPLING_FACTOR)

        count_tries = 0
        task = []
        while len(task) < self.task_size:
            count_tries += 1
            building = nearest_buildings[randint(0, len(nearest_buildings) - 1)]
            if not self._intersects_involved(building):
                task.append(building)
                self.involved_polygons.append(building.polygon)

            if count_tries > self.task_size * TaskGenerator.TRYING_FACTOR:
                return self._generate_mixed_task()

        return task

    def _generate_sequential_task(self):
        anchor_idx = randint(0, self.collection.length - 1)
        anchor = self.collection[anchor_idx].center
        buildings = self.collection.get_nearest(anchor, self.task_size * 2)

        task = []

        for i in range(len(buildings)):
            if len(task) == self.task_size:
                return task

            poly1 = buildings[i].polygon
            is_good_polygon = True

            for j in range(i):
                poly2 = buildings[j].polygon

                if poly1.intersects(poly2):
                    is_good_polygon = False
                    break
            if is_good_polygon:
                task.append(buildings[i])

        return task

    def _intersects_involved(self, building):
        for involved_polygon in self.involved_polygons:
            if involved_polygon.intersects(building.polygon):
                return True
        return False


def main(parsed_args):
    collection = Collection(parsed_args.data)

    directory = parsed_args.out_dir
    if not exists(directory):
        makedirs(directory)

    task_generator = TaskGenerator(collection, output_dir=directory, task_size=parsed_args.task_size)
    tasks = task_generator.generate(parsed_args.n_mix, parsed_args.n_seq)
    for i, task in enumerate(tasks):
        with open(directory + '/task' + str(i) + '.json', 'w') as file_desc:
            file_desc.write('[\n')
            for j, building in enumerate(task):
                building.flush(file_desc)
                if j < len(task) - 1:
                    file_desc.write(',\n')
            file_desc.write('\n]\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='path to shape file with markup')
    parser.add_argument('--out_dir', required=True, help='directory for generation')
    parser.add_argument('--task_size', required=True, type=int, help='task size')
    parser.add_argument('--n_mix', required=True, type=int, help='number of mixed tasks')
    parser.add_argument('--n_seq', required=True, type=int, help='number of sequential tasks')
    args = parser.parse_args()

    main(args)