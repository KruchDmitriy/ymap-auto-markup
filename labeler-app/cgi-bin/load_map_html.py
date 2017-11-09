#!/usr/bin/env python3


def load_map_html(script_name):
    map_html = """
    <html>
    <head>
        <title>Карты</title>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
        <script src="https://api-maps.yandex.ru/2.1/?loag=package.full&lang=ru_RU" type="text/javascript"></script>
        <script src="https://yandex.st/jquery/1.8.0/jquery.min.js" type="text/javascript"></script>
        <script src="/{}" type="text/javascript"></script>
    </head>
    <body>
        <div id="map" style="position:fixed; width:100%; height:100%; top:0px; left:0px;"/>
    </body>
    </html>
    """.format(script_name)

    return map_html
