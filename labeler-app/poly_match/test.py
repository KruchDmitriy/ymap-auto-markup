import utm
import numpy as np
from matplotlib import pyplot as plt
from poly_match import Polygon, Point, Edge, distance, grad_descent, \
                        find_affine, OptimizationParamsBuilder, AffineTransform
from affine import PolyTransform, AffineMx


def get_poly_center(poly):
    return np.mean(poly[:-1], axis=0)

def draw_polys(p1, p2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(p1[:, 0], p1[:, 1], color='blue', linewidth=3, solid_capstyle='round', zorder=2)
    ax.plot(p2[:, 0], p2[:, 1], color='red', linewidth=3, solid_capstyle='round', zorder=2)
    ax.set_aspect(1)
    plt.show()

def test_rotate():
    points = np.array([
            [0, 0], [0, 1], [1, 1], [1, 0], [0, 0]
        ], dtype=np.float64)

    poly = Polygon(points)
    poly.rotate(np.pi / 4)

    points = PolyTransform.rotate(points, np.pi / 4)

    print(poly.get_points())
    print(points)

def get_test_poly():
    poly = [
        [
            1.29892427362,
            8.2021771477
        ],
        [
            3.15222207333,
            7.66761609694
        ],
        [
            2.7278708638,
            6.9812176112
        ],
        [
            2.63060833204,
            6.6765852509
        ],
        [
            1.79680122907,
            6.9798907681
        ],
        [
            1.70088945682,
            6.9396833088
        ],
        [
            1.29573604808,
            7.0336899118
        ],
        [
            1.51143613844,
            7.5236542376
        ],
        [
            1.31312530217,
            7.5808346705
        ],
        [
            1.42308431317,
            7.8318419424
        ],
        [
            1.1917801254,
            7.8657050163
        ],
        [
            1.29892427362,
            8.2021771477
        ]
    ]

    return poly

def test():
    # points = np.array([
    #             [0, 0], [0, 1], [1, 1], [1, 0], [0, 0]
    #         ], dtype=np.float64)
    points = get_test_poly()
    poly1 = np.array(points, dtype=np.float64)

    orig_theta = 0.4
    orig_scale = 1.2

    poly2 = PolyTransform.rotate(poly1, orig_theta)
    poly2 = PolyTransform.scale(poly2, orig_scale)

    draw_polys(poly1, poly2)

    result = find_affine(Polygon(poly1), Polygon(poly2), OptimizationParamsBuilder().build())
    trans = result.transform
    print('theta ' + str(trans.theta))
    print('scale ' + str(trans.scale))
    print('shift ' + str((trans.shift_x, trans.shift_y)))
    print('residual ' + str(result.residual))
    theta = trans.theta
    scale = trans.scale
    shift = np.array((trans.shift_x, trans.shift_y))

    poly_res = PolyTransform.translate_vec(poly1, -get_poly_center(poly1))

    poly_res = PolyTransform.rotate(poly_res, theta)
    poly_res = PolyTransform.scale(poly_res, scale)
    poly_res = PolyTransform.translate_vec(poly_res, shift)

    poly_res = PolyTransform.translate_vec(poly_res, get_poly_center(poly2));

    draw_polys(poly_res, poly2)

def test_poly_distance():
    # points = np.array([
    #             [0, 0], [0, 1], [1, 1], [1, 0], [0, 0]
    #         ], dtype=np.float64)
    points = np.array(get_test_poly(), dtype=np.float64)
    p = Polygon(points)
    print('points')
    print(points)

    print(np.min(points[:, 0]))

    xs = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 100)
    ys = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), 100)

    fun_map = np.empty((xs.size, ys.size))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            fun_map[j, i] = p.distance(Point(x, y))

    fig = plt.figure()
    s = fig.add_subplot(1, 1, 1, xlabel='$xs$', ylabel='$ys$')
    s.plot(points[:, 0], points[:, 1], linewidth=3, color='red')
    im = s.imshow(
        fun_map,
        extent=(xs[0], xs[-1], ys[0], ys[-1]),
        origin='lower')
    fig.colorbar(im)
    plt.show()

def test_distance():
    u = Point(0, 0)
    v = Point(1, 1)
    e = Edge(u, v)

    xs = np.linspace(-2, 2, 100)
    ys = np.linspace(-2, 2, 100)

    fun_map = np.empty((xs.size, ys.size))
    for x in range(xs.size):
        for y in range(ys.size):
            fun_map[x, y] = distance(Point(xs[x], ys[y]), e)

    fig = plt.figure()
    s = fig.add_subplot(1, 1, 1, xlabel='$xs$', ylabel='$ys$')
    s.plot([u.x, v.x], [u.y, v.y], linewidth=3, color='red')
    im = s.imshow(
        fun_map,
        extent=(xs[0], xs[-1], ys[0], ys[-1]),
        origin='lower')
    fig.colorbar(im)
    plt.show()

def test_grad_desc(centered=True):
    # points = np.array([
    #         [0, 0], [0, 1], [1, 1], [1, 0], [0, 0]
    #     ], dtype=np.float64)
    points = get_test_poly()
    poly1 = np.array(points, dtype=np.float64)

    if centered:
        center_x, center_y = get_poly_center(poly1)
        poly1 = PolyTransform.translate(poly1, -center_x, -center_y)

    for i in range(100):
        orig_theta = 0.1 * i + 2

        print("orig_theta " + str(orig_theta))
        orig_scale = 1.2
        poly2 = PolyTransform.rotate(poly1, orig_theta)
        poly2 = PolyTransform.scale(poly2, orig_scale)
        draw_polys(poly1, poly2)

        poly_res = poly1
        for j in range(20):
            transform = AffineTransform(0, 0, 0, 1)
            transform = grad_descent(Polygon(poly_res), Polygon(poly2), transform, 10, 0.001)
            print(transform.shift_x, transform.shift_y, transform.theta, transform.scale)

            poly_res = PolyTransform.translate(poly_res, transform.shift_x, transform.shift_y)
            poly_res = PolyTransform.scale(poly_res, transform.scale)
            poly_res = PolyTransform.rotate(poly_res, transform.theta)
            draw_polys(poly_res, poly2)

def test_trans_commutation():
    for i in range(1000):
        a1 = np.random.rand(4) * 100
        mx1 = AffineMx.trans_from_params((a1[0], a1[1]), a1[2], a1[3])
        trans1 = AffineTransform(a1[0], a1[1], a1[2], a1[3])

        a2 = np.random.rand(4) * 100
        mx2 = AffineMx.trans_from_params((a2[0], a2[1]), a2[2], a2[3])
        trans2 = AffineTransform(a2[0], a2[1], a2[2], a2[3])

        mx = np.matmul(mx2, mx1)
        trans1.merge(trans2)
        trans = trans1
        mx_trans = AffineMx.trans_from_params((trans.shift_x, trans.shift_y),
            trans.theta, trans.scale)

        residual = np.sum((mx - mx_trans)** 2.)
        if residual > 1e-10:
            print(residual)
            print(mx)
            print(mx_trans)

            print('param1' + str(a1))
            print('param2' + str(a2))
            input()


def test_trans():
    for i in range(1000):
        a = np.random.rand(4)
        mx = AffineMx.trans_from_params((a[0], a[1]), a[2], a[3])
        trans = AffineTransform(a[0], a[1], a[2], a[3])

        points = np.random.rand(10, 2)
        poly = Polygon(points)

        poly.transform(trans)
        points_poly = poly.get_points()

        points_affine = np.vstack(
                            (points.transpose(),
                            np.ones(points.shape[0]))
                        ).transpose()

        points_after = np.matmul(points_affine, mx.transpose())
        points_after = points_after[:,:-1]

        residual = np.sum((points_poly - points_after)** 2.)
        if residual > 0.0001:
            print(residual)
            print(points_poly_affine)
            print(points_after)
            input()

if __name__ == '__main__':
    test()
