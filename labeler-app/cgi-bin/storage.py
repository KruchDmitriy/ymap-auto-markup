import os
import json
import uuid


class Storage:
    USERS_FILE = 'data/users.json'
    COOKIES_FILE = 'data/cookies.json'

    def __init__(self):
        try:
            with open(self.USERS_FILE, 'r', encoding='utf-8'):
                pass
        except FileNotFoundError:
            with open(self.USERS_FILE, 'w', encoding='utf-8') as f:
                json.dump({}, f)

        try:
            with open(self.COOKIES_FILE, 'r', encoding='utf-8'):
                pass
        except FileNotFoundError:
            with open(self.COOKIES_FILE, 'w', encoding='utf-8') as f:
                json.dump({}, f)

    def find(self, user):
        with open(self.USERS_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)
        if user in users:
            return True
        return False

    def check_login(self, user, password):
        with open(self.USERS_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)
        if user in users and users[user] == password:
            return True
        return False

    def register(self, user, password):
        if self.find(user):
            return False

        with open(self.USERS_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)

        users[user] = password
        with open(self.USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f)
        return True

    def set_cookie(self, user):
        with open(self.COOKIES_FILE, 'r', encoding='utf-8') as f:
            cookies = json.load(f)

        cookies = {k: v for k, v in cookies.items() if v != user}

        cookie = str(uuid.uuid4())
        cookies[cookie] = user
        with open(self.COOKIES_FILE, 'w', encoding='utf-8') as f:
            json.dump(cookies, f)
        return cookie

    def find_cookie(self, cookie):
        with open(self.COOKIES_FILE, 'r', encoding='utf-8') as f:
            cookies = json.load(f)
        return cookies.get(cookie)
