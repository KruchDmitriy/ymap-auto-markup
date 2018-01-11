#pragma once

#include <vector>
#include <iostream>
#include <cmath>
#include "3rd-party/json.hpp"

#ifdef WITH_PYTHON
#include <boost/python/def.hpp>
#include <boost/python/module.hpp>
#include <boost/python/numpy.hpp>

namespace bp = boost::python;
namespace np = boost::python::numpy;
#endif

struct AffineTransform {
    double shift_x;
    double shift_y;
    double theta;
    double scale;

    AffineTransform()
    : shift_x(0.)
    , shift_y(0.)
    , theta(0.)
    , scale(0.) {}

    AffineTransform(double shift_x, double shift_y, double theta, double scale)
        : shift_x(shift_x)
        , shift_y(shift_y)
        , theta(theta)
        , scale(scale) {}

    void merge(const AffineTransform& transform) {
        const double sinT = sin(transform.theta);
        const double cosT = cos(transform.theta);
        const double x = shift_x;
        const double y = shift_y;
        shift_x = cosT * transform.scale * x
                + sinT * transform.scale * y + transform.shift_x;
        shift_y = -sinT * transform.scale * x
                + cosT * transform.scale * y + transform.shift_y;
        scale *= transform.scale;
        theta += transform.theta;
    }

    AffineTransform& operator+=(const AffineTransform& transform) {
        shift_x += transform.shift_x;
        shift_y += transform.shift_y;
        theta += transform.theta;
        scale += transform.scale;
        return *this;
    }

    friend AffineTransform operator+(AffineTransform lhs, const AffineTransform& rhs) {
        lhs += rhs;
        return lhs;
    }

    AffineTransform& operator*=(const double lambda) {
        shift_x *= lambda;
        shift_y *= lambda;
        theta *= lambda;
        scale *= lambda;
        return *this;
    }

    friend AffineTransform operator*(AffineTransform lhs, const double rhs) {
        lhs *= rhs;
        return lhs;
    }

    AffineTransform& operator/=(const double lambda) {
        shift_x /= lambda;
        shift_y /= lambda;
        theta /= lambda;
        scale /= lambda;
        return *this;
    }

    friend AffineTransform operator/(AffineTransform lhs, const double rhs) {
        lhs /= rhs;
        return lhs;
    }

    AffineTransform& operator-=(const AffineTransform& transform) {
        shift_x -= transform.shift_x;
        shift_y -= transform.shift_y;
        theta -= transform.theta;
        scale -= transform.scale;
        return *this;
    }

    friend AffineTransform operator-(AffineTransform lhs, const AffineTransform& rhs) {
        lhs -= rhs;
        return lhs;
    }

    friend std::ostream& operator<< (std::ostream& os, const AffineTransform& trans) {
        os << "dx: " << trans.shift_x << " ";
        os << "dy: " << trans.shift_y << " ";
        os << "theta: " << trans.theta << " ";
        os << "scale: " << trans.scale << " ";

        return os;
    }
};

struct AffineResult {
    AffineTransform transform;
    double residual;
};

struct Point {
    double x, y;
    Point() : x(0.), y(0.) {}
    Point(double x, double y) : x(x), y(y) {}
};

class Edge {
private:
    Point _first;
    Point _second;
public:
    Edge(Point first, Point second)
        : _first(first), _second(second) {}

    Point first() const {
        return _first;
    }

    Point second() const {
        return _second;
    }
};

struct OptimizationParams {
    double min_shift, max_shift;
    double min_theta, max_theta;
    double min_scale, max_scale;
    int grid_step;
    int desc_num_steps;
    double desc_lr;

    OptimizationParams(
        double min_shift, double max_shift,
        double min_theta, double max_theta,
        double min_scale, double max_scale,
        int grid_step, int desc_num_steps, double desc_lr)
    : min_shift(min_shift)
    , max_shift(max_shift)
    , min_theta(min_theta)
    , max_theta(max_theta)
    , min_scale(min_scale)
    , max_scale(max_scale)
    , grid_step(grid_step)
    , desc_num_steps(desc_num_steps)
    , desc_lr(desc_lr) {}
};


struct OptimizationParamsBuilder {
private:
    OptimizationParams params;
public:
    OptimizationParamsBuilder()
    : params(-0.1, 0.1, -M_PI, M_PI, 0.7, 1.3, 4, 10, 0.01)
    {}

    void set_min_shift(double min_shift) {
        params.min_shift = min_shift;
    }

    void set_max_shift(double max_shift) {
        params.max_shift = max_shift;
    }

    void set_min_theta(double min_theta) {
        params.min_theta = min_theta;
    }

    void set_max_theta(double max_theta) {
        params.max_theta = max_theta;
    }

    void set_min_scale(double min_scale) {
        params.min_scale = min_scale;
    }

    void set_max_scale(double max_scale) {
        params.max_scale = max_scale;
    }

    void set_grid_step(int grid_step) {
        params.grid_step = grid_step;
    }

    void set_desc_num_steps(int desc_num_steps) {
        params.desc_num_steps = desc_num_steps;
    }

    void set_learn_rate(double rate) {
        params.desc_lr = rate;
    }

    OptimizationParams build() {
        return params;
    }
};

class Polygon {
private:
    std::vector<Point> points;
    Point _center;
    void calc_center();
public:
    Polygon(const std::vector<Point>& points);
    Polygon(nlohmann::json& json);
    double distance(const Point& p) const;
    void translate(double dx, double dy);
    void rotate(double theta);
    void scale(double scale);
    void transform(const AffineTransform& transform);
    Point closest_point(const Point& p) const;
    AffineTransform grad(const Polygon& poly, bool is_direct) const;
    Point center() const;

    const std::vector<Point>& get_points() const {
        return points;
    }

    #ifdef WITH_PYTHON
    Polygon(const np::ndarray& points) {
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

    np::ndarray as_np_array() const {
        bp::tuple shape = bp::make_tuple(points.size(), 2);
        bp::tuple stride = bp::make_tuple(sizeof(double) * 2, sizeof(double));
        return np::from_data(points.data(), np::dtype::get_builtin<double>(),
            shape, stride, bp::object());
    }
    #endif
};

AffineResult find_affine(Polygon poly_real, Polygon poly_pred, const OptimizationParams& opt_params);
