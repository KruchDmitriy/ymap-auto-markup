#!/usr/bin/env python3

import socket

if __name__ == "__main__":
    print("Content-Type: application/json\n")

    client = socket.socket()
    client.connect(('localhost', 9090))

    result = client.recv(1024)
    print(str(result, 'utf-8'))
    client.close()
