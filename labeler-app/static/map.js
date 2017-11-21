ymaps.ready(init);

var map, objects, polygons = [], index = {};
var marker = undefined;

const defaultColor = '#ffff00';
const goodColor = '#0000ff';
const badColor = '#ff0000';
const selectedColor = '#ffffff';
const selectedGoodColor = '#bbbbff';
const selectedBadColor = '#ffbbbb';

function init() {
    ymaps.geocode('Нижний Новгород', { results: 1 }).then(function (res) {
        var firstGeoObject = res.geoObjects.get(0);
        map = new ymaps.Map("map", {
            center: firstGeoObject.geometry.getCoordinates(),
            type: "yandex#hybrid",
            zoom: 18,
            controls: ['zoomControl']
        });

        map.container.fitToViewport();

        $("#next-task-button").click(function(event) {
            nextTask();
            event.stopPropagation();
        });

        $("#go-to-markup").click(function(event) {
            map.setBounds(map.geoObjects.getBounds());
            event.stopPropagation();
        });

        map.events.add('boundschange', function (e) {
            if (e.get('newZoom') !== e.get('oldZoom')) {
                if (marker !== null) {
                    if (e.get('newZoom') > 14) {
                        map.geoObjects.remove(marker)
                    }
                    else {
                        map.geoObjects.add(marker)
                    }
                }
            }
        });

        loadMarkup();
    }, function(err) {
        console.error(err.message);
    });

    const KEY_CODES = {
        "ENTER": 13,
        "LEFT_ARROW": 37,
        "UP_ARROW": 38,
        "RIGHT_ARROW": 39,
        "DOWN_ARROW": 40,
        "Y": 89,
        "N": 78,
        "ESCAPE": 27,
        "ZOOM_IN": 187,
        "ZOOM_OUT": 189
    };

    window.addEventListener('keydown', function(event) {
        switch (event.keyCode) {
            case KEY_CODES["LEFT_ARROW"]:
                prevPolygon();
                break;
            case KEY_CODES["RIGHT_ARROW"]:
                nextPolygon();
                break;
            case KEY_CODES["UP_ARROW"]: case KEY_CODES["Y"]:
                if (current === undefined)
                    break;

                mark(false, false);
                break;
            case KEY_CODES["DOWN_ARROW"]: case KEY_CODES["N"]:
                if (current === undefined)
                    break;

                mark(true, false);
                break;
            case KEY_CODES["ESCAPE"]:
                cancel();
                checkedPolygonIdx = undefined;
                break;
            case KEY_CODES["ENTER"]:
                nextTask();
                break;
            case KEY_CODES["ZOOM_IN"]:
                if (event.ctrlKey) {
                    map.setZoom(map.getZoom() + 1);
                }
                break;
            case KEY_CODES["ZOOM_OUT"]:
                if (event.ctrlKey) {
                    map.setZoom(map.getZoom() - 1);
                }
                break;
            default:
                return;
        }

        event.stopPropagation();
        event.preventDefault();
    });
}

function allChecked() {
        if (objects === undefined)
            return false;

        for (var i = 0; i < objects.length; i++) {
            if (objects[i].isBad === undefined) {
                return false;
            }
        }

        return true;
    }

var current = null;
var before = null;

function setColor(c) {
    if (c !== current) {
        if (objects[c].isBad != null) {
            polygons[c].options.set('fillColor', objects[c].isBad ? badColor : goodColor);
        }
        else {
            polygons[c].options.set('fillColor', defaultColor);
        }
    }
    else {
        if (objects[c].isBad != null) {
            polygons[c].options.set('fillColor', objects[c].isBad ? selectedBadColor : selectedGoodColor);
        }
        else {
            polygons[c].options.set('fillColor', selectedColor);
        }
    }
}

function select(c) {
    if (current !== null) {
        deselect()
    }
    current = c;
    before = objects[current].isBad;
    setColor(c);
    $('#dialog').show();
}

function deselect() {
    var c = current;
    current = null;
    setColor(c);
    $('#dialog').hide();
}

function cancel() {
    if (current === null) {
        return;
    }
    objects[current].isBad = before;
    deselect(current);
}

function nextPolygon() {
    var next;
    if (current === undefined) {
        next = 0;
    } else {
        next = (current + 1) % polygons.length;
    }

    select(next);
    map.setBounds(polygons[current].geometry.getBounds());
    map.setZoom(19);
}

function prevPolygon() {
    var next;
    if (current === undefined) {
        next = polygons.length - 1;
    } else {
        next = (current + polygons.length - 1) % polygons.length;
    }

    select(next);
    map.setBounds(polygons[current].geometry.getBounds());
    map.setZoom(19);
}

function mark(isBad, de=true) {
    if (current === null) {
        return;
    }
    objects[current].isBad = isBad;
    setColor(current);
    saveMarkup(current);
    if (de) {
        deselect();
    }
}

$(document).ready(function(){
    $("#yes").click(function(){
        mark(false);
    });
    $("#no").click(function(){
        mark(true);
    });
    $("#dialog").draggable();
});

function saveMarkup(id) {
    $.ajax({
        type: "POST",
        url: "/map/save_data",
        contentType: "application/json;charset=UTF-8",
        data: JSON.stringify({
            id: objects[id].id,
            isBad: objects[id].isBad
        }),
        dataType: "json"
    });
}

function nextTask() {
    if (!allChecked()) {
        alert("Проверьте, пожалуйста, всю разметку. Отмечайте, также и хорошую разметку (она должна загореться синим цветом).");
    } else {
        $.ajax({
            type: "POST",
            url: "/map/save_data",
            contentType: "application/json;charset=UTF-8",
            data: JSON.stringify({
                complete: "da"
            }),
            dataType: "json"
        });
        loadMarkup();
    }
}

function removeChildren(element) {
    while (element.firstChild) {
        element.removeChild(element.firstChild);
    }
}

function loadMarkup() {
    map.geoObjects.removeAll();

    $.post('/map/get_data', {}).done(function(data) {
        if (data === null) {
            $(location).attr('href', '/finish');
            return;
        }
        objects = data.task;

        const num_all = $("#num_all")[0];
        const num_finished = $("#num_finished")[0];
        removeChildren(num_all);
        removeChildren(num_finished);

        num_all.appendChild(document.createTextNode(data.available));
        num_finished.appendChild(document.createTextNode(data.done));
        index = {};

        for (var i = 0; i < objects.length; i++) {
            if (i === 0) {
                marker = new ymaps.Placemark(objects[i].coords, {
                     iconCaption: "Разметка"
                })
            }

            index[objects[i].id] = i;

            polygons[i] = new ymaps.Polygon([objects[i].coords], {}, {
                fillColor: defaultColor,
                strokeColor: "#000000",
                strokeWidth: 2,
                fillOpacity: 0.5
            });
            polygons[i].idx = i;
            polygons[i].events.add('click', function(e) {
                select(e.get('target').idx)
            });

            map.geoObjects.add(polygons[i]);
        }

        for (i in data.results) {
            var id = data.results[i].id;
            objects[index[id]].isBad = data.results[i].isBad;
            polygons[index[id]].options.set('fillColor', data.results[i].isBad ? badColor : goodColor);
        }

        map.setBounds(map.geoObjects.getBounds());
    });
}




