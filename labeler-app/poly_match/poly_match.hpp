#pragma once

#include <vector>
#include <iostream>
#include <math.h>
#include "3rd-party/json.hpp"

#ifdef WITH_PYTHON
#include <boost/python/def.hpp>
#include <boost/python/module.hpp>
#include <boost/python/numpy.hpp>

namespace bp = boost::python;
namespace np = boost::python::numpy;
#endif

#define sqr(a) ((a)*(a))

struct OptimizationParams {
  double min_shift, max_shift;
  double min_theta, max_theta;
  double min_scale, max_scale;
  int grid_step;
  int desc_num_steps;
  double desc_lr;
  double reg_rate;

  OptimizationParams(
          double min_shift, double max_shift,
          double min_theta, double max_theta,
          double min_scale, double max_scale,
          int grid_step, int desc_num_steps,
          double desc_lr, double reg_rate)
          : min_shift(min_shift)
          , max_shift(max_shift)
          , min_theta(min_theta)
          , max_theta(max_theta)
          , min_scale(min_scale)
          , max_scale(max_scale)
          , grid_step(grid_step)
          , desc_num_steps(desc_num_steps)
          , desc_lr(desc_lr)
          , reg_rate(reg_rate) {}
};

struct Point {
  double x, y;
  Point() : x(0.), y(0.) {}
  Point(double x, double y) : x(x), y(y) {}

  double distance(const Point& other) const {
    return sqr(x - other.x) + sqr(y - other.y);
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
  Point center() const;

  int size() const {
    return (int)points.size();
  }

  const Point& vertex(int idx) const {
    return points[idx];
  }

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

std::ostream& operator <<(std::ostream& out, const Polygon& poly) {
  out << "[";
  for (int i = 0; i < poly.size(); i++) {
    if (i > 0)
      out << ", ";
    out << "[" << poly.vertex(i).x << "," << poly.vertex(i).y <<  "]";
  }
  out << "]";
  return out;
}

struct AffineTransform {
  double shift_x;
  double shift_y;
  double theta;
  double scale;
  double c_x;
  double c_y;

  AffineTransform()
          : shift_x(0.)
          , shift_y(0.)
          , theta(0.)
          , scale(1.)
          , c_x(0.)
          , c_y(0.) {}

  AffineTransform(double shift_x, double shift_y, double theta, double scale, double c_x, double c_y)
          : shift_x(shift_x)
          , shift_y(shift_y)
          , theta(theta)
          , scale(scale)
          , c_x(c_x)
          , c_y(c_y) {}

  double gradStep(const Polygon& src, const Polygon& dst, double step, double lambda);

  AffineTransform& operator+=(const AffineTransform& transform) {
    shift_x += transform.shift_x;
    shift_y += transform.shift_y;
    theta += transform.theta;
    scale += transform.scale;
    c_x += c_x;
    c_y += c_y;
    return *this;
  }

  double regularization() const {
    return shift_x * shift_x
           + shift_y * shift_y
           + theta * theta
           + (1. - scale) * (1. - scale);
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
    c_x *= c_x;
    c_y *= c_y;
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
    c_x /= c_x;
    c_y /= c_y;
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
    c_x -= c_x;
    c_y -= c_y;
    return *this;
  }

  friend AffineTransform operator-(AffineTransform lhs, const AffineTransform& rhs) {
    lhs -= rhs;
    return lhs;
  }

  friend std::ostream& operator<< (std::ostream& os, const AffineTransform& trans) {
    os << "dx: " << trans.shift_x << " ";
    os << "dy: " << trans.shift_y << " ";
    os << "th: " << trans.theta << " ";
    os << "sc: " << trans.scale << " ";
    os << "cx: " << trans.c_x << " ";
    os << "cy: " << trans.c_y << " ";

    return os;
  }

  Polygon transform(const Polygon& polygon) const {
    const double sinT = sin(theta);
    const double cosT = cos(theta);
    std::vector<Point> points;

    for (const Point& p: polygon.get_points()) {
      double px = p.x;
      double py = p.y;
      Point transformed;
      transformed.x = cosT * scale * (px - c_x) + c_x
                      + sinT * scale * (py - c_y) + c_y
                      + shift_x;

      transformed.y = -sinT * scale * (px - c_x) + c_x
                      + cosT * scale * (py - c_y) + c_y
                      + shift_y;
      points.push_back(transformed);
    }
    return Polygon(points);
  }
};

struct AffineResult {
  AffineTransform transform;
  double residual;
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

struct OptimizationParamsBuilder {
private:
  OptimizationParams params;
public:
  OptimizationParamsBuilder()
          : params(-0.1, 0.1, -M_PI / 4., M_PI / 4., 0.7, 1.3, 4, 1000000, 1e-3, 0.)
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

  void set_reg_rate(double rate) {
    params.reg_rate = rate;
  }

  OptimizationParams build() {
    return params;
  }
};

AffineResult find_affine(Polygon poly_real, Polygon poly_pred, const OptimizationParams& opt_params);
