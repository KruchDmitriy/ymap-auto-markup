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
  int desc_num_steps;
  double desc_lr;
  double reg_rate;

  OptimizationParams(int desc_num_steps, double desc_lr, double reg_rate)
          : desc_num_steps(desc_num_steps)
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

  AffineTransform()
          : shift_x(0.)
          , shift_y(0.)
          , theta(0.)
          , scale(1.) {}

  AffineTransform(double shift_x, double shift_y, double theta, double scale)
          : shift_x(shift_x)
          , shift_y(shift_y)
          , theta(theta)
          , scale(scale) {}

  double gradStep(const Polygon& src, const Polygon& dst, double step, double lambda);

  AffineTransform& operator+=(const AffineTransform& transform) {
    shift_x += transform.shift_x;
    shift_y += transform.shift_y;
    theta += transform.theta;
    scale += transform.scale;
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

  Polygon transform(const Polygon& polygon) const {
    const double sinT = sin(theta);
    const double cosT = cos(theta);
    std::vector<Point> points;

    for (const Point& p: polygon.get_points()) {
      double px = p.x;
      double py = p.y;
      Point transformed;
      transformed.x = cosT * scale * px
                      + sinT * scale * py
                      + shift_x;

      transformed.y = -sinT * scale * px
                      + cosT * scale * py
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
  OptimizationParamsBuilder() : params(100000, 1e-3, 0.) {}

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
