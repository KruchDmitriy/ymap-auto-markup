import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


from pycocotools.coco import COCO
from pycocotools import mask

from osgeo import gdal, osr
import numpy as np
import json
import sys


PATH_TO_IMG = "../data/processedBuildingLabels/3band/"
PATH_TO_PREPROCESSED_IMG = "../preprocessedData/imgs/"
PATH_TO_LABELS = "../preprocessedData/labels/"
PATH_TO_GEOJSON = "../data/processedBuildingLabels/vectordata/geojson/"

####################
# download spacenet utilities from:
#  https://github.com/SpaceNetChallenge/utilities/tree/master/python/spaceNet
path_to_spacenet_utils = 'utilities/python'
sys.path.extend([path_to_spacenet_utils])
from spaceNetUtilities import geoTools as gT

###############################################################################
def geojson_to_pixel_arr(raster_file, geojson_file, pixel_ints=True,
                       verbose=False):
    '''
    Tranform geojson file into array of points in pixel (and latlon) coords
    pixel_ints = 1 sets pixel coords as integers
    '''

    # load geojson file
    with open(geojson_file) as f:
        geojson_data = json.load(f)

    # load raster file and get geo transforms
    src_raster = gdal.Open(raster_file)
    targetsr = osr.SpatialReference()
    targetsr.ImportFromWkt(src_raster.GetProjectionRef())

    geom_transform = src_raster.GetGeoTransform()

    # get latlon coords
    latlons = []
    types = []
    for feature in geojson_data['features']:
        coords_tmp = feature['geometry']['coordinates'][0]
        type_tmp = feature['geometry']['type']
        if verbose:
            print("features:", feature.keys())
            print("geometry:features:", feature['geometry'].keys())

            #print "feature['geometry']['coordinates'][0]", z
        latlons.append(coords_tmp)
        types.append(type_tmp)
        #print feature['geometry']['type']

    # convert latlons to pixel coords
    pixel_coords = []
    latlon_coords = []
    for i, (poly_type, poly0) in enumerate(zip(types, latlons)):

        if poly_type.upper() == 'MULTIPOLYGON':
            #print "oops, multipolygon"
            for poly in poly0:
                poly=np.array(poly)
                if verbose:
                    print("poly.shape:", poly.shape)

                # account for nested arrays
                if len(poly.shape) == 3 and poly.shape[0] == 1:
                    poly = poly[0]

                poly_list_pix = []
                poly_list_latlon = []
                if verbose:
                    print("poly", poly)
                for coord in poly:
                    if verbose:
                        print("coord:", coord)
                    lon, lat, z = coord
                    px, py = gT.latlon2pixel(lat, lon, input_raster=src_raster,
                                         targetsr=targetsr,
                                         geom_transform=geom_transform)
                    poly_list_pix.append([px, py])
                    if verbose:
                        print("px, py", px, py)
                    poly_list_latlon.append([lat, lon])

                if pixel_ints:
                    ptmp = np.rint(poly_list_pix).astype(int)
                else:
                    ptmp = poly_list_pix
                pixel_coords.append(ptmp)
                latlon_coords.append(poly_list_latlon)

        elif poly_type.upper() == 'POLYGON':
            poly=np.array(poly0)
            if verbose:
                print("poly.shape:", poly.shape)

            # account for nested arrays
            if len(poly.shape) == 3 and poly.shape[0] == 1:
                poly = poly[0]

            poly_list_pix = []
            poly_list_latlon = []
            if verbose:
                print("poly", poly)
            for coord in poly:
                if verbose:
                    print("coord:", coord)
                lon, lat, z = coord
                px, py = gT.latlon2pixel(lat, lon, input_raster=src_raster,
                                     targetsr=targetsr,
                                     geom_transform=geom_transform)
                poly_list_pix.append([px, py])
                if verbose:
                    print("px, py", px, py)
                poly_list_latlon.append([lat, lon])

            if pixel_ints:
                ptmp = np.rint(poly_list_pix).astype(int)
            else:
                ptmp = poly_list_pix
            pixel_coords.append(ptmp)
            latlon_coords.append(poly_list_latlon)

        else:
            print("Unknown shape type in coords_arr_from_geojson()")
            return

    return pixel_coords, latlon_coords


def main():
    for file in os.listdir(PATH_TO_IMG):
        img = cv2.imread(PATH_TO_IMG + file)

        geo_name = "Geo" + file[5:-4] + '.geojson'

        pixel_coords, latlon_coords = geojson_to_pixel_arr(
            PATH_TO_IMG + file,
            PATH_TO_GEOJSON + geo_name)

        file_name = file[:-4]

        img = cv2.imread(PATH_TO_IMG + file)
        cv2.imwrite(PATH_TO_PREPROCESSED_IMG + file_name + '.png', img)

        dst = np.zeros(shape=(img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
        cv2.fillPoly(dst, pixel_coords, color=(255, 255, 255))
        cv2.imwrite(PATH_TO_LABELS + file_name + '.png', dst)


if __name__ == "__main__":
    main()
