ymaps.ready(init);
var map, objects, polygons = [];
var checkedPolygonIdx = undefined;

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

        const loadButton = new ymaps.control.Button({
            data: {
                content: "Следующее задание"
            },
            options: {
                maxWidth: 250
            }
        });

        const centerButton = new ymaps.control.Button({
            data: {
                content: "Переместиться к разметке"
            },
            options: {
                maxWidth: 250
            }
        });

        map.container.fitToViewport();
        map.controls.add(loadButton, {
            float: 'right'
        });

        map.controls.add(centerButton, {
           float: 'right'
        });

        loadButton.events.add('click', function(event) {
            nextTask();
            event.originalEvent.target.state.set('selected', true);
            event.stopPropagation();
        });
        centerButton.events.add('click', function(event) {
            centerMapView();

            event.originalEvent.target.state.set('selected', true);
            event.stopPropagation();
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

    function loadMarkup() {
        removePolygons();

        $.post('/map/get_data', {}).done(function(data) {
            if (data === null) {
                $(location).attr('href', '/finish');
                return;
            }
            objects = data.task
            index = {}

            for (var i = 0; i < objects.length; i++) {
                index[objects[i].id] = i
                polygons[i] = new ymaps.Polygon([objects[i].coords[0]], {}, {
                    fillColor: defaultColor,
                    strokeColor: "#000000",
                    strokeWidth: 2,
                    fillOpacity: 0.5,
                    balloonContentLayout: ymaps.templateLayoutFactory.createClass(
                            '<h3 id="layout-element">Верна ли разметка?</h3>' +
                            '<button id="btn-yes">Да</button>' +
                            '<button id="btn-no">Нет</button>' +
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
                                    checkedPolygonIdx = $("#input_object_id").get(0).value;
                                    markPolygonAsGood(checkedPolygonIdx);
                                },

                                onNoButton: function() {
                                    checkedPolygonIdx = $("#input_object_id").get(0).value;
                                    markPolygonAsBad(checkedPolygonIdx);
                                }
                            }
                        )
                });
                map.geoObjects.add(polygons[i]);
            }

            for (i in data.results) {
                var id = data.results[i].id
                objects[index[id]].isBad = data.results[i].isBad
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

    function markPolygonAsGood(id) {
        polygons[id].options.set('fillColor', goodColor);
        objects[id].isBad = false;
        polygons[id].balloon.close();
        saveMarkup(id);
    }

    function markPolygonAsBad(id) {
        polygons[id].options.set('fillColor', badColor);
        objects[id].isBad = true;
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
    }

    function prevPolygon() {
        if (checkedPolygonIdx === undefined) {
            checkedPolygonIdx = polygons.length - 1;
        } else {
            polygons[checkedPolygonIdx].balloon.close();
            checkedPolygonIdx = (checkedPolygonIdx + polygons.length - 1) % polygons.length;
        }

        polygons[checkedPolygonIdx].balloon.open();
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

                markPolygonAsGood(checkedPolygonIdx);
                break;
            case KEY_CODES["DOWN_ARROW"]: case KEY_CODES["N"]:
                if (checkedPolygonIdx === undefined)
                    break;

                markPolygonAsBad(checkedPolygonIdx);
                break;
            case KEY_CODES["ESCAPE"]:
                polygons[checkedPolygonIdx].balloon.close();
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
