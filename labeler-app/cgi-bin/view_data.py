#!/usr/bin/env python3

import fiona
import cgi
import logging
import socket
import json

from load_map_html import load_map_html
from daemon import Daemon


class DataLoader(Daemon):
    def __init__(self, path_to_data, **kwargs):
        super(DataLoader, self).__init__(**kwargs)
        self.path_to_data = path_to_data

    def run(self):
        collection = fiona.collection(self.path_to_data)
        logging.info(collection.schema)

        iterator = iter(collection)

        server = socket.socket()
        server.bind(('localhost', 9090))
        server.listen(1)

        while True:
            connection, address = server.accept()
            try:
                obj = json.JSONEncoder().encode(next(iterator))
                print(len(obj))
                connection.send(bytes(obj, 'utf-8'))
                connection.close()
            except StopIteration:
                break
        server.close()
        self.stop()


if __name__ == "__main__":
    form = cgi.FieldStorage()
    path = form.getfirst("path_to_data", "../data/yandex/bld_sample.shp")

    loader = DataLoader(path_to_data=path,
                        pidfile="./cgi-bin/data/daemon_loader.cfg",
                        stdout="./cgi-bin/data/daemon_out.txt",
                        stderr="./cgi-bin/data/daemon_out.txt")

    print("Content-type: text/html\n")
    print(load_map_html("load_data.js"))

    loader.start()
