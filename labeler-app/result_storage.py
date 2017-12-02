import json


class ResultStorage:
    RESULTS_PATH = "./data/results.json"

    def __init__(self, results_path):
        if results_path is None:
            results_path = ResultStorage.RESULTS_PATH

        with open(results_path, 'r') as results_fp:
            self.results = json.load(results_fp)
            checked_buildings = {}

            for task in self.results:
                for bld_markup in task['results']:
                    bld_id = bld_markup['id']
                    if bld_id not in checked_buildings:
                        checked_buildings[bld_id] = not bld_markup['isBad']
                    else:
                        checked_buildings[bld_id] = checked_buildings[bld_id] or (not bld_markup['isBad'])

            self.checked_bld_idx = list(map(lambda pair: pair[0], filter(lambda pair: pair[1], checked_buildings.items())))
