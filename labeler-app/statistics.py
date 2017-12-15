import argparse
import json
from os import makedirs
from os.path import exists
from data_storage import TaskManager


class Statistics:
    def __init__(self, path_to_results):
        self.task_manager = TaskManager()
        self.task_to_users = {}
        self.bld_to_check = {}
        self.user_stat = {}
        self.num_alone_tasks = 0
        self.transform_stat = {
            'original': {},
            'rotate': {},
            'trans': {},
            'scale': {},
            'point_shift': {}
        }

        for transform in self.transform_stat:
            self.transform_stat[transform]['numGood'] = 0
            self.transform_stat[transform]['numBad'] = 0

        with open(path_to_results, 'r') as results_file:
            self.results = json.load(results_file)

    @staticmethod
    def _get_bld_from_task(task, bld_id):
        for bld in task:
            if bld["id"] == bld_id:
                return bld

    def _process_check_by_task(self, task_id, markup_check, user):
        task = self.task_manager.get_task_by_id(task_id)
        for markup in markup_check:
            markup_id = markup["id"]
            bld = Statistics._get_bld_from_task(task, markup_id)
            meta = "original"

            plus_to = "numGood"
            if markup["isBad"]:
                plus_to = "numBad"

            if "meta" in bld:
                meta = bld["meta"]

                for transform in meta["transform"]:
                    self.transform_stat[transform][plus_to] += 1
            else:
                self.transform_stat['original'][plus_to] += 1

            payload = {
                "user": user,
                "isBad": markup["isBad"]
            }

            if markup_id not in self.bld_to_check:
                self.bld_to_check[markup_id] = {}

            if task_id not in self.bld_to_check[markup_id]:
                self.bld_to_check[markup_id][task_id] = {
                    "coords": bld["coords"],
                    "meta": meta,
                    "markup": []
                }

            self.bld_to_check[markup_id][task_id]["markup"].append(payload)

    def _process_check_by_user(self, check, user):
        if user not in self.user_stat:
            self.user_stat[user] = {
                "numCompletedTasks": 0,
                "numCheckedBlds": 0,
                "numAloneCheckedTasks": 0
            }

        self.user_stat[user]["numCompletedTasks"] += 1

        for _ in check:
            self.user_stat[user]["numCheckedBlds"] += 1

    def _calc_coherence(self):
        for user in self.user_stat:
            self.user_stat[user]["numDisagreement"] = 0

        for bld in self.bld_to_check:
            for task_id in self.bld_to_check[bld]:
                self._calc_coherence_markup(
                    self.bld_to_check[bld][task_id]["markup"]
                )

    def _calc_coherence_markup(self, markup):
        bld_users = list(set(map(lambda check: check['user'], markup)))

        if len(bld_users) >= 2:
            user_to_mark = {}
            for check in markup:
                user_to_mark[check['user']] = check['isBad']

            for user1 in bld_users:
                for user2 in bld_users:
                    if user_to_mark[user1] != user_to_mark[user2]:
                        self.user_stat[user1]["numDisagreement"] += 1
                        self.user_stat[user2]["numDisagreement"] += 1
        else:
            self.num_alone_tasks += 1
            self.user_stat[bld_users[0]]["numAloneCheckedTasks"] += 1

    def calculate(self):
        for result in self.results:
            user = result['user']
            task = result['task']

            if task not in self.task_to_users:
                self.task_to_users[task] = [user]
            else:
                self.task_to_users[task].append(user)

            check = result['results']
            self._process_check_by_task(task, check, user)
            self._process_check_by_user(check, user)
        self._calc_coherence()

    @staticmethod
    def _json_dump(obj, filename):
        with open(filename, "w") as f:
            json.dump(obj, f, sort_keys=True, indent=4, separators=(',', ': '))

    def dump(self, out_dir):
        if not exists(out_dir):
            makedirs(out_dir)

        print("Number of alone checked tasks " + str(self.num_alone_tasks))

        Statistics._json_dump(self.task_to_users, out_dir + "/task_to_users.json")
        Statistics._json_dump(self.bld_to_check, out_dir + "/bld_to_check.json")
        Statistics._json_dump(self.user_stat, out_dir + "/user_stat.json")
        Statistics._json_dump(self.transform_stat, out_dir + "/transform_stat.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='path to results.json')
    parser.add_argument('--out_dir', required=True, help='path to output directory')

    args = parser.parse_args()

    statistics = Statistics(args.data)
    statistics.calculate()
    statistics.dump(args.out_dir)
