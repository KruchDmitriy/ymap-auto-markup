#!/usr/bin/env python3

import cgi
import html
from storage import Storage


storage = Storage()

form = cgi.FieldStorage()
action = form.getfirst("action", "")

login = form.getfirst("login", "")
login = html.escape(login)
password = form.getfirst("password", "")
password = html.escape(password)

if storage.find(login) and not storage.check_login(login, password):
    print("Content-type: text/html\n")
    print("""
        <!DOCTYPE HTML>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Обработка данных формы</title>
        </head>
        <body>
            <h1>Ooops...</h1>
            Ошибка при вводе логина или пароля
        </body>
        </html>""")
else:
    if not storage.find(login):
        storage.register(login, password)
    cookie = storage.set_cookie(login)
    print("Set-cookie: session={}".format(cookie))

    print("Content-type: text/html\n")
    with open("map.html", "r") as f:
        for line in f.readlines():
            print(line)
