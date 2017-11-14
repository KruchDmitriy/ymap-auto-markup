from os import makedirs, listdir
from os.path import exists, abspath


class TaskManager:
    def __init__(self):
        self.MARKUP_TASKS = "./data/markup_tasks/"
        if not exists(self.MARKUP_TASKS):
            makedirs(self.MARKUP_TASKS)

        self.tasks_for_user = {}
        self.user_counter = {}
        self.tasks = {}

        self.num_tasks = 0
        for filename in listdir(self.MARKUP_TASKS):
            with open(self.MARKUP_TASKS + filename, 'r') as file:
                self.tasks[filename] = file.read()
                self.num_tasks += 1
        if len(self.tasks) == 0:
            print(["No tasks found in the", abspath(self.MARKUP_TASKS), "directory!"])
            exit(-1)

    def start_user(self, user):
        return

    def next_task(self, user):
        if user not in self.user_counter.keys():
            self.user_counter[user] = 0
            self.tasks_for_user[user] = list(self.tasks.values())
        else:
            self.user_counter[user] += 1

        if self.user_counter[user] >= self.num_tasks:
            return None

        return self.tasks_for_user[user][self.user_counter[user]]


if __name__ == '__main__':
    taskManager = TaskManager()
    print(taskManager.next_task('vasya'))
    print(taskManager.next_task('vasya'))
    print(taskManager.next_task('vasya'))
    print(taskManager.next_task('vasya'))
    print(taskManager.next_task('vasya'))
