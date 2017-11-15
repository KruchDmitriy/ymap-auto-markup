ymaps.ready(init);

var map, objects, polygons = [];
var marker = undefined;
var checkedPolygonIdx = undefined;
var currentPolygon = undefined;

const defaultColor = '#ffff00';
const goodColor = '#0000ff';
const badColor = '#ff0000';

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
            centerMapView();
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

    function removePolygons() {
        map.geoObjects.removeAll();
    }

    function centerMapView() {
        map.setBounds(map.geoObjects.getBounds());
    }

    function removeChildren(element) {
        while (element.firstChild) {
            element.removeChild(element.firstChild);
        }
    }

    function loadMarkup() {
        removePolygons();

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
                    marker = new ymaps.Placemark(objects[i].coords[0][0], {
                         iconCaption: "Разметка"
                    })
                }

                index[objects[i].id] = i;

                polygons[i] = new ymaps.Polygon([objects[i].coords[0]], {}, {
                    fillColor: defaultColor,
                    strokeColor: "#000000",
                    strokeWidth: 2,
                    fillOpacity: 0.5,
                    balloonContentLayout: ymaps.templateLayoutFactory.createClass(
                            '<h6>Верна ли разметка?</h6>' +
                            '<div style="text-align: center">' +
                            '<button id="btn-yes" class="btn-success" ' +
                                'style="margin: 5%;' +
                                'width: 40%;">' +
                            'Да' +
                            '</button>' +
                            '<button id="btn-no" class="btn-danger" ' +
                                'style="margin: 5%;' +
                                'width: 40%">' +
                            'Нет' +
                            '</button>' +
                            '</div>' +
                            '<input type="hidden" id="input_object_id" value="' + i + '"/>', {
                                build: function() {
                                    this.constructor.superclass.build.call(this);
                                    $('#btn-yes').bind('click', this.onYesButton);
                                    $('#btn-no').bind('click', this.onNoButton);
                                },

                                clear: function() {
                                    $('#btn-yes').unbind('click', this.onYesButton);
                                    $('#btn-no').unbind('click', this.onNoButton);
                                    this.constructor.superclass.clear.call(this);
                                },

                                onYesButton: function() {
                                    markPolygon(checkedPolygonIdx, false);
                                },

                                onNoButton: function() {
                                    markPolygon(checkedPolygonIdx, true);
                                }
                            }
                        )
                });

                map.geoObjects.add(polygons[i]);
            }

            for (i in data.results) {
                var id = data.results[i].id;
                objects[index[id]].isBad = data.results[i].isBad;
                polygons[index[id]].options.set('fillColor', data.results[i].isBad ? badColor : goodColor);
            }

            centerMapView();
        });
    }

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

    function markPolygon(id, isBad) {
        polygons[id].options.set('fillColor', isBad ? badColor : goodColor);
        objects[id].isBad = isBad;
        polygons[id].balloon.close();
        saveMarkup(id);
    }

    function nextPolygon() {
        if (checkedPolygonIdx === undefined) {
            checkedPolygonIdx = 0;
        } else {
            polygons[checkedPolygonIdx].balloon.close();
            checkedPolygonIdx = (checkedPolygonIdx + 1) % polygons.length;
        }

        polygons[checkedPolygonIdx].balloon.open();
        currentPolygon = polygons[checkedPolygonIdx];
    }

    function prevPolygon() {
        if (checkedPolygonIdx === undefined) {
            checkedPolygonIdx = polygons.length - 1;
        } else {
            polygons[checkedPolygonIdx].balloon.close();
            checkedPolygonIdx = (checkedPolygonIdx + polygons.length - 1) % polygons.length;
        }

        polygons[checkedPolygonIdx].balloon.open();
        currentPolygon = polygons[checkedPolygonIdx];
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

    window.addEventListener('keydown', function(event) {
        switch (event.keyCode) {
            case KEY_CODES["LEFT_ARROW"]:
                nextPolygon();
                break;
            case KEY_CODES["RIGHT_ARROW"]:
                prevPolygon();
                break;
            case KEY_CODES["UP_ARROW"]: case KEY_CODES["Y"]:
                if (checkedPolygonIdx === undefined)
                    break;

                markPolygon(checkedPolygonIdx, false);
                break;
            case KEY_CODES["DOWN_ARROW"]: case KEY_CODES["N"]:
                if (checkedPolygonIdx === undefined)
                    break;

                markPolygon(checkedPolygonIdx, true);
                break;
            case KEY_CODES["ESCAPE"]:
                currentPolygon.balloon.close();
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
