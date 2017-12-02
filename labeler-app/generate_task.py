import fiona
import json
import argparse
import utm
import numpy as np
from sys import exit
from os import makedirs
from os.path import exists
from scipy.spatial import KDTree
from numpy.random import randint
from shapely.geometry import Polygon
from result_storage import ResultStorage
from copy import deepcopy


class Collection:
    EPSILON = 1e-7

    """
        Constructor of class collection
        
        :param
        buildings -- list of Building instances
    """
    def __init__(self, buildings):
        self.sorted_storage = sorted(buildings,
                                     key=lambda k: [k.raw_coords[0][0], k.raw_coords[0][1]])
        self._create_index()
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

    def _create_index(self):
        self.index = {}

        for i, bld in enumerate(self.sorted_storage):
            self.index[bld.id] = i

    def get_by_bld_id(self, id):
        return self.sorted_storage[self.index[id]]


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
        return center_point.xy[0][0], center_point.xy[1][0]

    @property
    def center_utm(self):
        center_point = self.polygon.centroid.coords
        utm_center = utm.from_latlon(longitude=center_point.xy[0][0],
                                     latitude=center_point.xy[1][0])
        return utm_center[0], utm_center[1]

    def apply_transform(self, transform):
        for i, point in enumerate(self.raw_coords):
            self.raw_coords[i] = transform(point)
        self.polygon = Polygon(self.raw_coords)

    def flush(self, file_desc):
        bld_json = {
            'id': self.id,
            'coords': self.raw_coords,
        }

        json.dump(bld_json, file_desc)


class TaskGenerator:
    SAMPLING_FACTOR = 5
    TRYING_FACTOR = 10

    def __init__(self, collection, variator=None):
        self.variator = variator
        self.collection = collection
        self.tasks = []
        self.involved_polygons = []

    def generate(self, task_size, num_mixed, num_sequential):
        tasks = []
        for i in range(num_mixed):
            tasks.append(self._generate_mixed_task(task_size))

        for i in range(num_sequential):
            tasks.append(self._generate_sequential_task(task_size))

        if self.variator is not None:
            num_methods = np.random.poisson(size=len(tasks))
            num_methods = list(map(int, num_methods))

            variator_methods_length = len(Variator.METHODS)
            for i in range(len(tasks)):
                methods = []
                for counter in range(min(4, num_methods[i])):
                    method = Variator.METHODS[np.random.randint(0, variator_methods_length - 1)]
                    methods.append(method)

                tasks[i] = self.variator.apply(tasks[i], methods=methods)

        return tasks

    def _generate_mixed_task(self, task_size):
        anchor_idx = randint(0, self.collection.length - 1)
        anchor = self.collection[anchor_idx].center

        nearest_buildings = self.collection.get_nearest(anchor, task_size * TaskGenerator.SAMPLING_FACTOR)

        count_tries = 0
        task = []
        while len(task) < task_size:
            count_tries += 1
            building = nearest_buildings[randint(0, len(nearest_buildings) - 1)]
            if not self._intersects_involved(building):
                task.append(building)
                self.involved_polygons.append(building.polygon)

            if count_tries > task_size * TaskGenerator.TRYING_FACTOR:
                return self._generate_mixed_task(task_size)

        return task

    def _generate_sequential_task(self, task_size):
        anchor_idx = randint(0, self.collection.length - 1)
        anchor = self.collection[anchor_idx].center
        buildings = self.collection.get_nearest(anchor, task_size * 2)

        task = []

        for i in range(len(buildings)):
            if len(task) == task_size:
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


class Variator:
    METHODS = ['rotate', 'trans', 'scale', 'point_shift']
    _ID_MATRIX = np.eye(3)

    def __init__(self, sigma_trans, sigma_rotate, sigma_scale):
        self.rng_rotate = lambda: np.random.normal(0., sigma_rotate)
        self.rng_trans = lambda: np.random.normal([0., 0.], sigma_trans)
        self.rng_scale = lambda: np.random.normal(1., sigma_scale)

    @staticmethod
    def _affine_transform(point, affine_matrix):
        src_utm = utm.from_latlon(longitude=point[0], latitude=point[1])

        src = np.array([src_utm[0], src_utm[1], 1.])
        dst = np.matmul(affine_matrix, src)

        dst_latlon = utm.to_latlon(dst[0], dst[1], src_utm[2], src_utm[3])
        return [dst_latlon[1], dst_latlon[0]]

    @staticmethod
    def _point_shift(point, rng):
        return point + rng()

    @staticmethod
    def _rotate(theta, center):
        rotation = np.array([
            [np.cos(theta), np.sin(theta), 0.],
            [-np.sin(theta), np.cos(theta), 0.],
            [0., 0., 1.]
        ])

        trans_to = Variator._trans(-center[0], -center[1])
        trans_from = Variator._trans(center[0], center[1])

        return np.matmul(np.matmul(trans_from, rotation), trans_to)

    @staticmethod
    def _trans(dx, dy):
        return np.array([
            [1., 0., dx],
            [0., 1., dy],
            [0., 0., 1.]
        ])

    @staticmethod
    def _scale(sigma, center):
        scale = np.array([
            [sigma, 0., 0.],
            [0., sigma, 0.],
            [0., 0., 1.]
        ])

        trans_to = Variator._trans(-center[0], -center[1])
        trans_from = Variator._trans(center[0], center[1])

        return np.matmul(np.matmul(trans_from, scale), trans_to)

    def apply(self, task, methods=METHODS, name=None):
        if name is not None:
            print(name)
        new_task = []

        for bld in task:
            rotation = Variator._ID_MATRIX
            translation = Variator._ID_MATRIX
            scaling = Variator._ID_MATRIX

            if 'rotate' in methods:
                rotation = Variator._rotate(self.rng_rotate(), bld.center_utm)

            if 'trans' in methods:
                trans = self.rng_trans()
                translation = Variator._trans(trans[0], trans[1])

            if 'scale' in methods:
                scaling = Variator._scale(self.rng_scale(), bld.center_utm)

            affine_matrix = np.matmul(np.matmul(rotation, translation), scaling)

            bld_copy = deepcopy(bld)
            bld_copy.apply_transform(
                lambda point: Variator._affine_transform(point, affine_matrix)
            )

            if 'point_shift' in methods:
                bld_copy.apply_transform(
                    lambda point: Variator._point_shift(point, self.rng_trans)
                )

            new_task.append(bld_copy)

        return new_task


def generate_tasks(task_generator, out_dir, task_size, num_mixed, num_sequential):
    tasks = task_generator.generate(task_size, num_mixed, num_sequential)
    for i, task in enumerate(tasks):
        with open(out_dir + '/task' + str(i) + '.json', 'w') as file_desc:
            file_desc.write('[\n')
            for j, building in enumerate(task):
                building.flush(file_desc)
                if j < len(task) - 1:
                    file_desc.write(',\n')
            file_desc.write('\n]\n')


def variated_task_generator(collection, results, shift, theta, scale):
    result_storage = ResultStorage(results)
    checked_buildings = []
    for bld_id in result_storage.checked_bld_idx:
        checked_buildings.append(collection.get_by_bld_id(bld_id))

    checked_collection = Collection(checked_buildings)
    return TaskGenerator(checked_collection, Variator(shift, theta, scale))


def main(parsed_args):
    out_dir = parsed_args.out_dir
    if not exists(out_dir):
        makedirs(out_dir)

    buildings = list(map(lambda item: Building(item),
                         fiona.collection(parsed_args.data)))
    collection = Collection(buildings)

    if parsed_args.variate:
        task_generator = variated_task_generator(collection, parsed_args.results, parsed_args.shift,
                                                 parsed_args.theta, parsed_args.scale)
    else:
        task_generator = TaskGenerator(collection)

    generate_tasks(task_generator, out_dir, parsed_args.task_size,
                   parsed_args.n_mix, parsed_args.n_seq)


def test():
    buildings = list(map(lambda item: Building(item),
                         fiona.collection('./data/yandex/bld_sample.shp')))
    collection = Collection(buildings)

    task = list(collection.get_nearest(collection.centers[0], 10))

    shifter = Variator(0.75, 0, 0)
    rotator = Variator(0, 0.05, 0)
    scaler = Variator(0, 0, 0.1)

    tasks = {
        'task_orig': task,
        'task_shift': task + shifter.apply(task, name='shift'),
        'task_rotate1': task + rotator.apply(task, name='rotate1'),
        'task_rotate2': task + rotator.apply(task, name='rotate2'),
        'task_scale': task + scaler.apply(task, name='scale')
    }

    for task_name, task_content in tasks.items():
        with open('./data/test/' + task_name + '.json', 'w') as file_desc:
            file_desc.write('[\n')
            for j, building in enumerate(task_content):
                building.flush(file_desc)
                if j < len(task_content) - 1:
                    file_desc.write(',\n')
            file_desc.write('\n]\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='path to shape file with markup')
    parser.add_argument('--out_dir', required=True, help='directory for generation')

    parser.add_argument('--task_size', required=True, type=int, help='task size')
    parser.add_argument('--n_mix', required=True, type=int, help='number of mixed tasks')
    parser.add_argument('--n_seq', required=True, type=int, help='number of sequential tasks')

    parser.add_argument('--variate', action='store_true', help='generate variation of markup')
    parser.add_argument('--results', help='path to results')
    parser.add_argument('--shift', type=float, help='std dev for variation by shift')
    parser.add_argument('--theta', type=float, help='std dev for variation by angle')
    parser.add_argument('--scale', type=float, help='std dev for variation by scale')

    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    if args.test:
        test()
        exit()

    main(args)
