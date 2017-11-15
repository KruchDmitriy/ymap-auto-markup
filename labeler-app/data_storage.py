import logging
from os import makedirs, listdir
from os.path import exists, abspath
from random import randint
from copy import deepcopy

import json

class TaskManager:
    MARKUP_TASKS = "./data/markup_tasks/"
    RESULTS = './data/results.json'

    def __init__(self):
        if not exists(self.MARKUP_TASKS):
            makedirs(self.MARKUP_TASKS)
        try:
            with open(self.RESULTS, 'r') as markup_file:
                self.results = json.load(markup_file)
        except FileNotFoundError:
            self.results = []

        self.tasks = {}
        for filename in listdir(self.MARKUP_TASKS):
            with open(self.MARKUP_TASKS + filename, 'r') as file:
                self.tasks[filename] = json.load(file)
        if not self.tasks:
            print(["No tasks found in the", abspath(self.MARKUP_TASKS), "directory!"])
            exit(-1)
        self.tasks_index = list(map(lambda task: {
                'task': task,
                'occupied': list(
                    map(lambda result: result['user'],
                        filter(lambda result: result['task'] == task, self.results)))
            }, self.tasks.keys()))
        self.users_index = {}
        for t in self.tasks_index:
            tid = t['task']
            for u in t['occupied']:
                if u not in self.users_index:
                    self.users_index[u] = []
                self.users_index[u].append(tid)

    def append_result(self, user, task, markup_json):
        result = list(filter(lambda r: r['task'] == task and r['user'] == user, self.results))
        if result:
            result[0]['results'].append(markup_json)
            self._flush()
        else:
            logging.warning("Attempting to add result to unknown task: {} user: {}" % [task, user])

    def next_task(self, user):
        if user not in self.users_index:
            self.users_index[user] = []

        intersection = len(list(filter(
            lambda task: len(task['occupied']) > 1 and (user in task['occupied']),
            self.tasks_index)
        ))
        if intersection / (len(self.users_index[user]) + 0.5) < 0.1:
            task = self.next_intersection(user)
        else:
            task = self.next_empty()
        if task:
            taskId = task['task']
            self.results.append({
                'task': taskId,
                'user': user,
                'results': []
            })
            task['occupied'].append(user)
            self.users_index[user].append(taskId)
            self._flush()
            return taskId

    def next_intersection(self, user):
        possible_tasks = list(filter(lambda task: not (user in task['occupied']) and len(task['occupied']) > 0, self.tasks_index))
        if not possible_tasks:
            return self.next_empty()
        return possible_tasks[randint(0, len(possible_tasks) - 1)]

    def next_empty(self):
        possible_tasks = list(filter(lambda task: len(task['occupied']) == 0, self.tasks_index))
        if possible_tasks:
            return possible_tasks[randint(0, len(possible_tasks) - 1)]

    def task_by_id(self, task, user):
        if task in self.tasks:
            state = list(filter(lambda r: r['task'] == task and r['user'] == user, self.results))
            return {
                'task': deepcopy(self.tasks[task]),
                'results': state[0]['results'] if state else []
            }

    def _flush(self):
        with open(self.RESULTS, 'w') as markup_file:
            json.dump(self.results, markup_file)

