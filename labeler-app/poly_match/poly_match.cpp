#include <iostream>
#include <limits>
#include <iomanip>
#include "poly_match.hpp"

#ifdef WITH_PYTHON
namespace bp = boost::python;
namespace np = boost::python::numpy;
#endif

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
        double g_theta = ((p2.x - p1.x) * p2.y - (p2.y - p1.y) * p2.x);
        double g_scale = (p2.x - p1.x) * p2.x + (p2.y - p1.y) * p2.y;
        return { g_x, g_y, g_theta, g_scale };
    }

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

    AffineTransform grad_descent(Polygon poly_real, const Polygon& poly_pred,
                                const AffineTransform& transform,
                                const int num_steps, const double learning_rate) {
        AffineTransform cum_trans = transform;
        poly_real.transform(transform);
        for (int i = 0; i < num_steps; i++) {
            AffineTransform grad_direct = poly_real.grad(poly_pred, true);
            AffineTransform grad_reverse = poly_pred.grad(poly_real, false);

            AffineTransform cur_trans {0., 0., 0., 1.};
            cur_trans -= grad_direct * learning_rate / log(2. + i);
            cur_trans -= grad_reverse * learning_rate / log(2. + i);

            cum_trans.merge(cur_trans);

            poly_real.transform(cur_trans);
        }

        return cum_trans;
    }

    double normalize_theta(double theta) {
        return atan2(sin(theta), cos(theta));
    }
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

AffineResult find_affine(Polygon poly_real, Polygon poly_pred, const OptimizationParams& opt_params) {
    Point center1 = poly_real.center();
    Point center2 = poly_pred.center();

    poly_real.translate(-center1.x, -center1.y);
    poly_pred.translate(-center2.x, -center2.y);

    const double min_shift = opt_params.min_shift;
    const double max_shift = opt_params.max_shift;
    const double shift_step = (max_shift - min_shift) / opt_params.grid_step;

    const double min_theta = opt_params.min_theta;
    const double max_theta = opt_params.max_theta;
    const double theta_step = (max_theta - min_theta) / opt_params.grid_step;

    const double min_scale = opt_params.min_scale;
    const double max_scale = opt_params.max_scale;
    const double scale_step = (max_scale - min_scale) / opt_params.grid_step;

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

        Polygon tmp_poly = poly_real;
        tmp_poly.transform(transform);

        double res = utils::residual(tmp_poly, poly_pred);

        if (res < min_res) {
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
            utils::normalize_theta(best_theta), best_scale
        } , min_res };
}

void Polygon::calc_center() {
    _center = {0, 0};

    for (uint32_t i = 0; i < this->points.size() - 1; i++) {
        const Point& point = this->points[i];
        _center.x += point.x;
        _center.y += point.y;
    }

    _center.x /= this->points.size() - 1;
    _center.y /= this->points.size() - 1;
}

Polygon::Polygon(const std::vector<Point>& points) {
    this->points = points;
    calc_center();
}

Polygon::Polygon(nlohmann::json& json) {
    points = std::vector<Point>(json.size());
    for (uint32_t i = 0; i < json.size(); i++) {
        points[i] = { json[i][0], json[i][1] };
    }
    calc_center();
}

#ifdef WITH_PYTHON
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
        .def("as_np_array", &Polygon::as_np_array);

    class_<OptimizationParamsBuilder>("OptimizationParamsBuilder", init<>())
        .def("set_min_shift", &OptimizationParamsBuilder::set_min_shift)
        .def("set_max_shift", &OptimizationParamsBuilder::set_max_shift)
        .def("set_min_theta", &OptimizationParamsBuilder::set_min_theta)
        .def("set_max_theta", &OptimizationParamsBuilder::set_max_theta)
        .def("set_min_scale", &OptimizationParamsBuilder::set_min_scale)
        .def("set_max_scale", &OptimizationParamsBuilder::set_max_scale)
        .def("set_grid_step", &OptimizationParamsBuilder::set_grid_step)
        .def("set_desc_num_steps", &OptimizationParamsBuilder::set_desc_num_steps)
        .def("set_learn_rate", &OptimizationParamsBuilder::set_learn_rate)
        .def("build", &OptimizationParamsBuilder::build);

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
#endif
