#include <iostream>
#include <limits>
#include <iomanip>
#include <cmath>
#include "poly_match.hpp"

namespace bp = boost::python;
namespace np = boost::python::numpy;

namespace utils {
    Point rotate_point(Point point, double sinT, double cosT) {
        double px = point.x;
        double py = point.y;
        point.x = cosT * px + sinT * py;
        point.y = -sinT * px + cosT * py;
        return {point.x, point.y};
    }

    AffineTransform grad_affine(Point p1, Point p2) {
        double g_x = (p2.x - p1.x);
        double g_y = (p2.y - p1.y);
        double g_theta = (p2.x - p1.x) * p2.y - (p2.y - p1.y) * p2.x;
        double g_scale = (p2.x - p1.x) * p2.x + (p2.y - p1.y) * p2.y;
        return { g_x, g_y, g_theta, g_scale };
    }

    const double EPSILON = 1e-10;

    double point_distance(const Point& u, const Point& v) {
        return sqrt((u.x - v.x) * (u.x - v.x) + (u.y - v.y) * (u.y - v.y));
    }

    static double dot(const Point& u, const Point& v) {
        return u.x * v.x + u.y * v.y;
    }

    Point find_closest_point(const Point& p, const Edge& e) {
        Point u = e.first();
        Point v = e.second();

        const double uv_dist = point_distance(u, v);
        if (uv_dist < 1e-10) {
            return v;
        }

        Point vec1 { p.x - u.x, p.y - u.y };
        Point vec2 { v.x - u.x, v.y - u.y };

        const double t = std::max(0., std::min(1.,
            dot(vec1, vec2) / uv_dist / uv_dist));
        const Point proj { u.x + t * (v.x - u.x), u.y + t * (v.y - u.y) };
        return proj;
    }

    double distance(const Point& p, const Edge& e) {
        return point_distance(p, find_closest_point(p, e));
    }

    static double residual(const Polygon& poly1, const Polygon& poly2) {
        const std::vector<Point>& points1 = poly1.get_points();
        const std::vector<Point>& points2 = poly2.get_points();

        double sum_dist = 0.;
        for (uint32_t i = 0; i < points1.size() - 1; i++) {
            sum_dist += poly2.distance(points1[i]);
        }

        for (uint32_t i = 0; i < points2.size() - 1; i++) {
            sum_dist += poly1.distance(points2[i]);
        }

        return sum_dist / (points1.size() + points2.size() - 2.);
    }

    AffineTransform grad_descent(const Polygon& poly_real, Polygon poly_pred,
                                const AffineTransform& transform,
                                int num_steps, const double learning_rate) {
        AffineTransform cum_trans = transform;
        poly_pred.transform(transform);
        for (int i = 0; i < num_steps; i++) {
            AffineTransform grad_direct = poly_pred.grad(poly_real, true);
            AffineTransform grad_reverse = poly_real.grad(poly_pred, false);

            AffineTransform cur_trans {0., 0., 0., 1.};
            cur_trans -= grad_direct * learning_rate;
            cur_trans -= grad_reverse * learning_rate;

            cum_trans.merge(cur_trans);

            poly_pred.transform(cur_trans);
        }

        return cum_trans;
    }
}

Polygon::Polygon(const np::ndarray& points) {
    double* data_ = reinterpret_cast<double*>(points.get_data());
    uint32_t length = points.shape(0);

    for (uint32_t i = 0; i < length; i++) {
        this->points.push_back({ data_[i * 2], data_[i * 2 + 1] });
    }

    _center = {0, 0};

    for (uint32_t i = 0; i < this->points.size() - 1; i++) {
        const Point& point = this->points[i];
        _center.x += point.x;
        _center.y += point.y;
    }

    _center.x /= this->points.size() - 1;
    _center.y /= this->points.size() - 1;
}

AffineTransform Polygon::grad(const Polygon& poly, bool is_direct) const {
    using utils::grad_affine;

    AffineTransform grad {0., 0., 0., 0.};
    for (uint32_t i = 0; i < points.size() - 1; i++) {
        Point p2 = poly.closest_point(points[i]);
        AffineTransform point_grad;
        if (is_direct) {
            point_grad = grad_affine(p2, points[i]);
        } else {
            point_grad = grad_affine(points[i], p2);
        }
        grad += point_grad;
    }

    grad /= (points.size() - 1.);

    return grad;
}

Point Polygon::closest_point(const Point& p) const {
    double dist = std::numeric_limits<double>::max();
    Point c_point;
    for (uint32_t i = 0; i < points.size() - 1; i++) {
        Point new_point = utils::find_closest_point(p, Edge(points[i], points[i + 1]));
        double new_dist = utils::point_distance(p, new_point);
        if (new_dist < dist) {
            dist = new_dist;
            c_point = new_point;
        }
    }

    return c_point;
}

double Polygon::distance(const Point& p) const {
    double dist = std::numeric_limits<double>::max();
    for (uint32_t i = 0; i < points.size() - 1; i++) {
        double new_dist = utils::distance(p, Edge(points[i], points[i + 1]));
        if (new_dist < dist) {
            dist = new_dist;
        }
    }

    return dist;
}

void Polygon::transform(const AffineTransform& transform) {
    const double sinT = sin(transform.theta);
    const double cosT = cos(transform.theta);

    for (Point& p: points) {
        double px = p.x;
        double py = p.y;

        p.x = cosT * transform.scale * px
            + sinT * transform.scale * py
            + transform.shift_x;

        p.y = -sinT * transform.scale * px
            + cosT * transform.scale * py
            + transform.shift_y;
    }
}

void Polygon::translate(double dx, double dy) {
    for (Point& point: points) {
        point.x += dx;
        point.y += dy;
    }
}

void Polygon::rotate(double theta) {
    const double sinT = sin(theta);
    const double cosT = cos(theta);

    for (Point& point: points) {
        point = utils::rotate_point(point, sinT, cosT);
    }
}

void Polygon::scale(double scale) {
    for (Point& point: points) {
        point.x *= scale;
        point.y *= scale;
    }
}

Point Polygon::center() const {
    return _center;
}

AffineResult find_affine(Polygon poly_real, Polygon poly_pred, OptimizationParams opt_params) {
    Point center1 = poly_real.center();
    Point center2 = poly_pred.center();

    poly_real.translate(-center1.x, -center1.y);
    poly_pred.translate(-center2.x, -center2.y);

    const double min_shift = opt_params.min_shift;
    const double max_shift = opt_params.max_shift;
    const double shift_step = (max_shift - min_shift) / opt_params.steps;

    const double min_theta = opt_params.min_theta;
    const double max_theta = opt_params.max_theta;
    const double theta_step = (max_theta - min_theta) / opt_params.steps;

    const double min_scale = opt_params.min_scale;
    const double max_scale = opt_params.max_scale;
    const double scale_step = (max_scale - min_scale) / opt_params.steps;

    double best_theta = 0.;
    double best_scale = 1.;
    double best_x = 0.;
    double best_y = 0.;
    double min_res = std::numeric_limits<double>::max();

    for (double shift_x = min_shift; shift_x < max_shift; shift_x += shift_step) {
    for (double shift_y = min_shift; shift_y < max_shift; shift_y += shift_step) {
    for (double theta   = min_theta; theta   < max_theta; theta   += theta_step) {
    for (double scale   = min_scale; scale   < max_scale; scale   += scale_step) {
        AffineTransform transform = {shift_x, shift_y, theta, scale};

        transform = utils::grad_descent(poly_real, poly_pred, transform,
            opt_params.desc_num_steps, opt_params.desc_lr);

        Polygon tmp_poly = poly_pred;
        tmp_poly.transform(transform);

        double res = utils::residual(poly_real, tmp_poly);

        if (res < min_res) {
            std::cout << "min_res " << min_res << std::endl;
            // std::cout << "shift_x " << transform.shift_x << std::endl;
            // std::cout << "shift_y " << transform.shift_y << std::endl;
            // std::cout << "theta " << transform.theta << std::endl;
            // std::cout << "scale " << transform.scale << std::endl;

            min_res = res;
            best_theta = transform.theta;
            best_scale = transform.scale;
            best_x = transform.shift_x;
            best_y = transform.shift_y;
        }
    }}}}

    return {{
            best_x,
            best_y,
            best_theta, best_scale
        } , min_res };
}

Polygon::Polygon(const std::vector<Point>& points) {
    this->points = points;

    _center = {0, 0};

    for (uint32_t i = 0; i < this->points.size() - 1; i++) {
        const Point& point = this->points[i];
        _center.x += point.x;
        _center.y += point.y;
    }

    _center.x /= this->points.size() - 1;
    _center.y /= this->points.size() - 1;
}

void test() {
    using namespace std;
    vector<Point> vec1 = { {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0} };
    Polygon poly1(vec1);

    vector<Point> vec2 = { {0, 0}, {0, 2}, {1, 2}, {1, 0}, {0, 0} };
    Polygon poly2(vec2);

    vector<Point> vec3 = {
        { 28.999129892427362, 41.04482021771477 },
        { 28.999315222207333, 41.044766761609694 },
        { 28.999272787086380, 41.04469812176112 },
        { 28.999263060833204, 41.04466765852509 },
        { 28.999179680122907, 41.04469798907681 },
        { 28.999170088945682, 41.04469396833088 },
        { 28.999129573604808, 41.04470336899118 },
        { 28.999151143613844, 41.04475236542376 },
        { 28.999131312530217, 41.04475808346705 },
        { 28.999142308431317, 41.04478318419424 },
        { 28.999119178012540, 41.04478657050163 },
        { 28.999129892427362, 41.04482021771477 }
    };
    Polygon poly3(vec3);

    vector<Point> vec4 = {
        { 699190.57115417, 3209614.5956814  },
        { 699180.98277655, 3209633.49578724 },
        { 699175.48857823, 3209627.43942008 },
        { 699172.82875896, 3209625.73641586 },
        { 699177.73971665, 3209617.36741134 },
        { 699177.59063786, 3209616.24421831 },
        { 699179.46947686, 3209612.06357219 },
        { 699183.60338552, 3209615.44358092 },
        { 699184.62919364, 3209613.42114555 },
        { 699186.74830375, 3209615.14683701 },
        { 699187.63264045, 3209612.7177504  },
        { 699190.57115417, 3209614.5956814  },
    };
    Polygon poly4(vec4);

    for (const Point& p1: poly1.get_points()) {
        cout << p1.x << " " << p1.y << endl;
    }

    cout << utils::residual(poly1, poly1) << endl;
    cout << utils::residual(poly1, poly2) << endl;
    cout << utils::residual(poly2, poly1) << endl;

    AffineResult result = find_affine(poly1, poly2, {
        -1., 1.,
        -M_PI, M_PI,
        0.7, 1.3,
        10,
        5,
        0.001
    });

    AffineTransform transform = result.transform;

    cout << "shift_x " << transform.shift_x << endl
        << "shift_y " << transform.shift_y << endl
        << "theta " << transform.theta << endl
        << "scale " << transform.scale << endl
        << "residual " << result.residual << endl;

    cout << utils::residual(poly3, poly4) << endl;

    result = find_affine(poly3, poly4, {
        -1., 1.,
        -M_PI, M_PI,
        0.7, 1.3,
        10,
        5,
        0.001
    });

    transform = result.transform;

    cout << "shift_x " << transform.shift_x << endl
        << "shift_y " << transform.shift_y << endl
        << "theta " << transform.theta << endl
        << "scale " << transform.scale << endl
        << "residual " << result.residual << endl;
}

void test_grad_desc() {
    std::vector<Point> points = {{-1, -1}, {-1, 1}, {1, 1}, {1, -1},{-1, -1}};
    Polygon poly1 = Polygon(points);

    double orig_theta = 0.2;
    double orig_scale = 1.2;
    Polygon poly2 = poly1;
    poly2.rotate(orig_theta);
    poly2.scale(orig_scale);

    for (int i = 0; i < 100; i++) {
        auto transform = AffineTransform(0, 0, 0, 1);
        transform = utils::grad_descent(Polygon(poly1), Polygon(poly2), transform, 10, 0.01);

        poly2.scale(transform.scale);
        poly2.rotate(transform.theta);
        poly2.translate(transform.shift_x, transform.shift_y);
    }
}

BOOST_PYTHON_MODULE(poly_match) {
    using namespace boost::python;

    np::initialize();

    class_<Point>("Point", init<double, double>())
        .add_property("x", &Point::x)
        .add_property("y", &Point::y);

    class_<Edge>("Edge", init<Point, Point>())
        .add_property("first", &Edge::first)
        .add_property("second", &Edge::second);

    def("find_affine", find_affine);
    def("test", test);
    def("residual", utils::residual);
    def("distance", utils::distance);
    def("grad_descent", utils::grad_descent);

    class_<Polygon>("Polygon", init<const np::ndarray&>())
        .def("distance" , &Polygon::distance)
        .def("translate", &Polygon::translate)
        .def("rotate"   , &Polygon::rotate)
        .def("scale"    , &Polygon::scale)
        .def("center"   , &Polygon::center)
        .def("transform", &Polygon::transform)
        .def("get_points", &Polygon::get_np_points);

    class_<OptimizationParams>("OptimizationParams",
        init<double, double, double, double, double, double, int, int, double>());

    class_<AffineTransform>("AffineTransform",
        init<double, double, double, double>())
        .def("merge", &AffineTransform::merge)
        .add_property("shift_x" , &AffineTransform::shift_x)
        .add_property("shift_y" , &AffineTransform::shift_y)
        .add_property("theta"   , &AffineTransform::theta)
        .add_property("scale"   , &AffineTransform::scale);

    class_<AffineResult>("AffineResult")
        .add_property("transform", &AffineResult::transform)
        .add_property("residual", &AffineResult::residual);
}

int main() {
    test_grad_desc();
    return 0;
}
