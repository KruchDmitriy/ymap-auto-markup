import json
from os import makedirs
from os.path import exists


class UserStorage:
    USERS_FILE = 'data/users.json'

    def __init__(self):
        if not exists('data'):
            makedirs('data')

        try:
            with open(self.USERS_FILE, 'r', encoding='utf-8'):
                pass
        except FileNotFoundError:
            with open(self.USERS_FILE, 'w', encoding='utf-8') as f:
                json.dump({}, f)

    def find(self, user):
        with open(self.USERS_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)
        if user in users:
            return True
        return False

    def get_password(self, user):
        with open(self.USERS_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)
        return users[user]

    def register(self, user, password):
        if self.find(user):
            return False

        with open(self.USERS_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)

        users[user] = password
        with open(self.USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f)
        return True
