import json
from os import makedirs
from os.path import exists
from flask_login import UserMixin
from passlib.hash import sha256_crypt


class UserStorage:
    USERS_FILE = 'data/users.json'

    def __init__(self):
        if not exists('data'):
            makedirs('data')

        try:
            with open(self.USERS_FILE, 'r', encoding='utf-8') as f:
                self.users = json.load(f)
        except FileNotFoundError:
            with open(self.USERS_FILE, 'w', encoding='utf-8') as f:
                json.dump({}, f)

    def find(self, user):
        if user in self.users:
            return User(self, self.users[user])
        return User(self)

    def register(self, user, password):
        if not self.find(user).is_anonymous:
            return False

        self.users[user] = {
            'id': user,
            'password': sha256_crypt.encrypt(password)
        }
        self._flush()
        return True

    def _flush(self):
        with open(self.USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.users, f)


class User(UserMixin):
    def __init__(self, storage, user=None):
        self.storage = storage
        self.json = user

    @property
    def is_authenticated(self):
        return self.json is not None

    @property
    def is_active(self):
        return self.json is not None

    @property
    def is_anonymous(self):
        return self.json is None

    def get_id(self):
        return self.json['id']

    def check_password(self, password):
        return sha256_crypt.verify(password, self.json['password'])

    def get_task(self):
        if 'task' in self.json:
            return self.json['task']

    def set_task(self, task):
        self.json['task'] = task
        self.storage._flush()
