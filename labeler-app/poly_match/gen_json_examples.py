import sys
sys.path.append('..')

import json
import numpy as np
from affine import PolyTransform


def get_test_poly():
    poly = [[44.736129324000046, 48.807937570000036],
            [44.736117816000046, 48.80792028800005],
            [44.736292752000054, 48.80786946400008],
            [44.73630261500006, 48.80788427600004],
            [44.73640551300008, 48.807854381000084],
            [44.73629263000004, 48.80768486800008],
            [44.73619910000008, 48.80771204100006],
            [44.73621840900006, 48.807741038000074],
            [44.73602190200006, 48.80779812900005],
            [44.73600136600004, 48.80776729000007],
            [44.73591244000005, 48.807793126000035],
            [44.736028195000074, 48.807966950000036],
            [44.736129324000046, 48.807937570000036]]

    return poly


def get_poly_center(poly):
    return np.mean(poly[:-1], axis=0)


def main():
    poly = np.array(get_test_poly())
    poly_center = get_poly_center(poly)
    poly = PolyTransform.translate_vec(poly, -poly_center)

    min_theta = -1e-3
    max_theta = 1e-3

    polys = [1] * 100
    for i in range(100):
        theta = min_theta + (max_theta - min_theta) / 100 * i
        polys[i] = PolyTransform.rotate(poly, theta)
        polys[i] = list(PolyTransform.translate_vec(polys[i], poly_center))
        for j, point in enumerate(polys[i]):
            polys[i][j] = list(point)

    with open('json_examples/gen.json', 'w') as f:
        json.dump(polys, f)

    poly = PolyTransform.translate_vec(poly, poly_center)
    poly = list(poly)
    for i, point in enumerate(poly):
        poly[i] = list(point)

    with open('json_examples/real.json', 'w') as f:
        json.dump([poly] * 100, f)

if __name__ == '__main__':
    main()
