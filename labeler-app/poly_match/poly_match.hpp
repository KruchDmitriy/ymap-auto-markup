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
    return sqrt(sqr(x - other.x) + sqr(y - other.y));
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

  void merge(const AffineTransform& transform, const OptimizationParams& params) {
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

    // clamp values
    // shift_x = std::min(params.max_shift, std::max(params.min_shift, shift_x));
    // shift_y = std::min(params.max_shift, std::max(params.min_shift, shift_y));
    scale = std::min(params.max_scale, std::max(params.min_scale, scale));
    theta = std::min(params.max_theta, std::max(params.min_theta, theta));
  }

  AffineTransform& operator+=(const AffineTransform& transform) {
    shift_x += transform.shift_x;
    shift_y += transform.shift_y;
    theta += transform.theta;
    scale += transform.scale;
    return *this;
  }
  
  

  AffineTransform grad_regularization() const {
    return { shift_x, shift_y, theta, 1. - scale };
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

  double gradStep(const Polygon& src, const Polygon& dst, double step, double lambda) {
    const Polygon& img = transform(src);

    double g_x = 0.;
    double g_y = 0.;
    double g_theta = 0.;
    double g_scale = 0.;

    double totalDistance = 0;

    for (int i = 0; i < img.size(); i++) {
      const Point& a_i = img.vertex(i);
      const Point& orig_a_i = src.vertex(i);
      Point closest;
      double distance = std::numeric_limits<double>::max();
      for (const Point& b_j: dst.get_points()) {
        if (distance > a_i.distance(b_j)) {
          distance = a_i.distance(b_j);
          closest = b_j;
        }
      }
      totalDistance += distance;
      {
        const double dist_x = a_i.x - closest.x;
        const double dist_y = a_i.y - closest.y;
        const double sinT = sin(theta);
        const double cosT = cos(theta);

        const double s = scale;

        g_x += dist_x;
        g_y += dist_y;
        g_theta += dist_x * (orig_a_i.x * s * (- sinT) + orig_a_i.y * s * cosT)
                  + dist_y * (orig_a_i.x * s * (- cosT) + orig_a_i.y * s * (- sinT));
        g_scale += dist_x * (orig_a_i.x * cosT + orig_a_i.y * sinT)
                  + dist_y * (orig_a_i.x * (- sinT) + orig_a_i.y * cosT);
      };
    }

    for (const Point& b_j: src.get_points()) {
      int closestIdx = 0;
      double distance = std::numeric_limits<double>::max();
      for (int i = 0; i < img.size(); i++) {
        const Point& a_i = img.vertex(i);
        if (distance > a_i.distance(b_j)) {
          distance = a_i.distance(b_j);
          closestIdx = i;
        }
      }
      totalDistance += distance;
      {
        const Point& closest = b_j;
        const Point& a_i = img.vertex(closestIdx);
        const Point& orig_a_i = src.vertex(closestIdx);

        const double dist_x = a_i.x - closest.x;
        const double dist_y = a_i.y - closest.y;
        const double sinT = sin(theta);
        const double cosT = cos(theta);

        const double s = scale;

        g_x += dist_x;
        g_y += dist_y;
        g_theta += dist_x * (orig_a_i.x * s * (- sinT) + orig_a_i.y * s * cosT)
                   + dist_y * (orig_a_i.x * s * (- cosT) + orig_a_i.y * s * (- sinT));
        g_scale += dist_x * (orig_a_i.x * cosT + orig_a_i.y * sinT)
                   + dist_y * (orig_a_i.x * (- sinT) + orig_a_i.y * cosT);
      }
    }

    const double g_module = sqrt(sqr(g_x) + sqr(g_y) + sqr(g_theta) + sqr(g_scale));
    shift_x -= (g_x + lambda * shift_x) * step;
    shift_y -= (g_y + lambda * shift_y) * step;
    theta -= (g_theta + lambda * theta) * step / 100;
    scale -= (g_scale + lambda * scale) * step / 100;
    return g_module;
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
          : params(-0.1, 0.1, -M_PI / 4., M_PI / 4., 0.7, 1.3, 4, 10000, 1e-2, 1e-4)
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
