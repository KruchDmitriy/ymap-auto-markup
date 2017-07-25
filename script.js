ymaps.ready(init);
var map, objects, polygons = [];
const markupDir = '/data/marked_out/';
const markupFileExt = '.json';
var fileCounter = -1;

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

        var loadButton = new ymaps.control.Button({
            data: {
                content: "Следующее задание"
            },
            options: {
                maxWidth: 250
            }
        });

        map.container.fitToViewport();
        map.controls.add(loadButton, {
            float: 'right'
        });

        loadButton.events.add('click', onTaskClick);
        loadMarkup();
    }, function(err) {
        console.error(err.message);
    });

    function allChecked() {
        if (objects == undefined)
            return false;

        for (var i = 0; i < objects.length; i++) {
            if (objects[i].isBad === "undefined") {
                return false;
            }
        }

        return true;
    }

    function getCurrentFileName() {
        return "task" + String(fileCounter) + markupFileExt;
    }

    function removePolygons() {
        map.geoObjects.removeAll();
        // for (var i = 0; i < polygons.length; i++) {
        //     map.geoObjects.
        // }
    }

    function loadMarkup() {
        removePolygons();

        fileCounter++;
        $.getJSON(markupDir + getCurrentFileName(), function(data) {
            objects = data;

            for (var i = 0; i < objects.length; i++) {
                polygons[i] = new ymaps.Polygon([objects[i].coords], {}, {
                    fillColor: defaultColor,
                    strokeColor: "#000000",
                    strokeWidth: 2,
                    fillOpacity: 0.5,
                    balloonContentLayout: ymaps.templateLayoutFactory.createClass(
                            '<h3 id="layout-element">Плохая разметка?</h3>' +
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

                                onYesButton: function(e) {
                                    var id = $("#input_object_id").get(0).value;
                                    polygons[id].options.set('fillColor', badColor);
                                    // polygons[id].options.set('strokeColor', badColor);
                                    objects[id].isBad = true;
                                    saveMarkup();
                                },

                                onNoButton: function(e) {
                                    var id = $("#input_object_id").get(0).value;
                                    polygons[id].options.set('fillColor', goodColor);
                                    // polygons[id].options.set('strokeColor', goodColor);
                                    objects[id].isBad = false;
                                    saveMarkup();
                                }
                            }
                        )
                });
                map.geoObjects.add(polygons[i]);
            }

            map.setBounds(map.geoObjects.getBounds());
        }).fail(function(err) {
            console.error("Error while loading markup");
        });
    }

    function saveMarkup() {
        $.ajax({
            type: "POST",
            url: "/cgi-bin/save.py",
            contentType: 'application/json',
            data: JSON.stringify({
                filename: getCurrentFileName(),
                data: objects
            }),
            dataType: 'json'
        });
    }

    function onTaskClick(event) {
        if (!allChecked()) {
            alert("Проверьте, пожалуйста, всю разметку. Отмечайте, также и хорошую разметку (она должна загореться синим цветом).");
        } else {
            loadMarkup();
        }

        event.originalEvent.target.state.set('selected', true);
        event.stopPropagation();
    }
}
