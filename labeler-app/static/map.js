ymaps.ready(init);

var map, objects, polygons = [], index = {}, taskId;
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
        map.events.add('boundschange', function (e) {
            if (e.get('newZoom') !== e.get('oldZoom')) {
                if (marker != undefined) {
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
}

function checkComplete() {
    var result = true;

    for (var i = 0; i < objects.length && result; i++) {
        // noinspection EqualityComparisonWithCoercionJS
        if (objects[i].isBad == undefined) {
            result = false;
        }
    }

    $("#next-task-button").toggleClass("disabled", !result);

    return result;
}

var selected = [];
var before = {};

function updateColor(c) {
    polygons[c].options.set('fillOpacity', "0.5");
    if (selected.indexOf(c) < 0) {
        if (objects[c].isBad != undefined) {
            polygons[c].options.set('fillColor', objects[c].isBad ? badColor : goodColor);
        }
        else {
            polygons[c].options.set('fillColor', defaultColor);
        }
    }
    else if (type !== 'single') {
        if (objects[c].isBad != undefined) {
            polygons[c].options.set('fillColor', objects[c].isBad ? selectedBadColor : selectedGoodColor);
        }
        else {
            polygons[c].options.set('fillColor', selectedColor);
        }
    }
    else {
        polygons[c].options.set('fillOpacity', "0");
    }
}

function select(c) {
    if (selected.indexOf(c) >= 0) {
        return;
    }
    selected.push(c);
    before[c] = objects[c].isBad;
    updateColor(c);
    $('#dialog').show();
}

function deselect(c) {
    var update = [];
    if (c == undefined) {
        update = selected;
        selected = []
    }
    else {
        var index = selected.indexOf(c);
        if (index >= 0) {
            selected.splice(index, 1);
            update = [c];
        }

    }
    for(var i in update) {
        updateColor(update[i]);
    }
    if (selected.length === 0)
        $('#dialog').hide();
}

function cancel() {
    if (selected == undefined) {
        return;
    }
    for (var i in selected) {
        objects[selected[i]].isBad = before[selected[i]];
    }
    deselect();
}

function nextPolygon() {
    var next;
    if (selected.length !== 1) {
        next = 0;
    } else {
        next = (selected[0] + 1) % polygons.length;
    }

    var index = 0;
    while (objects[next].isBad != undefined && index < polygons.length) {
        next = (next + 1) % polygons.length;
        index++;
    }

    deselect();
    select(next);
    var bounds = polygons[selected].geometry.getBounds();
    map.setCenter([(bounds[0][0] + bounds[1][0])/2, (bounds[0][1] + bounds[1][1])/2]);
    map.setZoom(19);
}

function prevPolygon() {
    var next;
    if (selected.length !== 1) {
        next = polygons.length - 1;
    } else {
        next = (selected[0] + polygons.length - 1) % polygons.length;
    }
    var index = 0;
    while (objects[next].isBad != undefined && index < polygons.length) {
        next = (next + polygons.length - 1) % polygons.length;
        index++;
    }

    deselect();
    select(next);
    var bounds = polygons[selected].geometry.getBounds();
    map.setCenter([(bounds[0][0] + bounds[1][0])/2, (bounds[0][1] + bounds[1][1])/2]);
    map.setZoom(19);
}

function mark(isBad) {
    if (selected == undefined) {
        return;
    }
    for (var i in selected) {
        objects[selected[i]].isBad = isBad;
        updateColor(selected[i]);
        saveMarkup(selected[i]);
    }
    checkComplete();
}

function saveMarkup(id) {
    $.ajax({
        type: "POST",
        url: "/map/save_data",
        contentType: "application/json;charset=UTF-8",
        data: JSON.stringify({
            'taskId': taskId,
            'id': objects[id].id,
            'isBad': objects[id].isBad
        }),
        dataType: "json"
    });
}

function nextTask() {
    if (!checkComplete()) {
        alert("Проверьте, пожалуйста, всю разметку. Отмечайте, также и хорошую разметку (она должна загореться синим цветом).");
        nextPolygon()
    } else {
        $.ajax({
            type: "POST",
            url: "/map/save_data",
            contentType: "application/json;charset=UTF-8",
            data: JSON.stringify({
                'complete': true
            }),
            dataType: "json",
            async: false
        });
        loadMarkup();
    }
}

function nMapsOpen() {
    var center = map.getCenter();
    var zoom = map.getZoom();
    var url = "https://n.maps.yandex.ru/#!/?z=" + zoom + "&ll=" + center[0] + "%2C" + center[1] + "&l=nk%23sat";
    window.open(url, '_blank');
}

function loadMarkup() {
    $.post('/map/get_data', {}).done(function(data) {
        if (data == undefined) {
            $(location).attr('href', '/finish');
            return;
        }
        taskId = data.id;
        objects = data.task;

        $("#num_all").text(data.available);
        $("#num_finished").text(data.done - 1);

        index = {};
        for (var i = 0; i < objects.length; i++) {
            index[objects[i].id] = i;
        }

        if (objects.length > 0) {
            marker = new ymaps.Placemark(objects[0].coords[0], {
                iconCaption: "Разметка"
            });
        }

        for (var i in data.results) {
            var id = data.results[i].id;
            if (index[id] == undefined || index[id] >= objects.length)
                continue;
            objects[index[id]].isBad = data.results[i].isBad;
        }

        draw();
        if (objects.length > 0) {
            map.setBounds(map.geoObjects.getBounds());
        }
        checkComplete()
    });
}

var type = "full";
function changeMarkupType(button) {
    if (type === "full") {
        type = "single";
        button.text('Вернуть разметку');
        draw(selected);
    }
    else {
        type = "full";
        button.text('Убрать разметку');
        draw();
    }
}

function draw(indices) {
    map.geoObjects.removeAll();
    polygons = [];
    for (var i = 0; i < objects.length; i++) {
        polygons[i] = new ymaps.Polygon([objects[i].coords], {}, {
            fillColor: defaultColor,
            strokeColor: "#000000",
            strokeWidth: 1,
            fillOpacity: 0.5
        });
        polygons[i].idx = i;
        if (type === "full") {
            polygons[i].events.add('click', function (e) {
                var index = e.get('target').idx;
                if (selected.indexOf(index) < 0) {
                    select(e.get('target').idx)
                }
                else {
                    deselect(e.get('target').idx)
                }
            });
        }
        updateColor(i);
        if (indices == undefined || indices.indexOf(i) >= 0) {
            map.geoObjects.add(polygons[i]);
        }
    }
}

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

$(document).ready(function(){
    $("#yes").click(function(){
        mark(false);
        deselect();
    });
    $("#no").click(function(){
        mark(true);
        deselect();
    });
    $("#dialog").draggable();
    $("#next-task-button").click(function(event) {
        nextTask();
        event.stopPropagation();
    });
    $("#go-to-markup").click(function(event) {
        map.setBounds(map.geoObjects.getBounds());
        event.stopPropagation();
    });
    $("#go-to-nmap").click(function(event) {
        nMapsOpen();
        event.stopPropagation();
    });
    $("#clear-screen").click(function(event) {
        changeMarkupType($(this));
        event.stopPropagation();
    });

    window.addEventListener('keydown', function(event) {
        switch (event.keyCode) {
            case KEY_CODES["LEFT_ARROW"]:
                prevPolygon();
                break;
            case KEY_CODES["RIGHT_ARROW"]:
                nextPolygon();
                break;
            case KEY_CODES["UP_ARROW"]: case KEY_CODES["Y"]:
                if (selected.length === 0)
                    break;

                mark(false);
                break;
            case KEY_CODES["DOWN_ARROW"]: case KEY_CODES["N"]:
                if (selected.length === 0)
                    break;

                mark(true);
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
});