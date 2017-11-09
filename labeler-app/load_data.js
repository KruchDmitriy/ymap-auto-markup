ymaps.ready(init);
var map, objects, polygons = [];

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

        var loadButton = new ymaps.control.Button({
            data: {
                content: "Загрузить данные"
            },
            options: {
                maxWidth: 250
            }
        });

        map.controls.add(loadButton, {
            float: 'right'
        });

        loadButton.events.add('click', function(event){ loadMarkup(); event.stopPropagation(); });
    }, function(err) {
        console.error(err.message);
    });

    function removePolygons() {
        map.geoObjects.removeAll();
    }

    function loadMarkup() {
        removePolygons();

        $.ajax({
            type: "POST",
            url: "/cgi-bin/load_data.py",
            contentType: 'application/json',
            success: function (response) {
                console.log(response);
                map.geoObjects.add(new ymaps.Polygon(response['geometry'], {}, {}));
                map.setBounds(map.geoObjects.getBounds());
            },
            error: function (xhr) {
                console.error(xhr.status + ": " + xhr.responseText);
                console.error("Error while loading markup");
            }
        });
    }
}
