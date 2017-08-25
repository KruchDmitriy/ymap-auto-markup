#!/usr/bin/env python3

import http.cookies
import os
from storage import Storage
import sys
import json
from pathlib import Path


storage = Storage()
cookie = http.cookies.SimpleCookie(os.environ.get("HTTP_COOKIE"))
session = cookie.get("session")
if session is not None:
    session = session.value
user = storage.find_cookie(session)

length = os.environ["CONTENT_LENGTH"]

content = sys.stdin.read(int(length))
with open('log.txt', 'w') as f:
    f.write(content)

content = json.loads(content, encoding='utf-8')

MARKUP_DIR = "data/checked/" + user + "/"
MARKUP_FILE = MARKUP_DIR + content['filename']
fd = Path(MARKUP_DIR)

del content['filename']
content['user'] = user

with open('log.txt', 'w') as f:
    json.dump(content, f)

if not fd.exists():
    os.mkdir(MARKUP_DIR)

with open(MARKUP_FILE, "w") as f:
    json.dump(content, f)
