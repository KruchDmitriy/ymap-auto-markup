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
    for (uint32_t i = 0; i < points1.size(); i++) {
      sum_dist += poly2.distance(points1[i]);
    }

    for (uint32_t i = 0; i < points2.size(); i++) {
      sum_dist += poly1.distance(points2[i]);
    }

    return sum_dist / (points1.size() + points2.size());
  }

  AffineTransform grad_descent(const Polygon& poly_real, const Polygon& poly_pred,
                               const AffineTransform& transform,
                               const OptimizationParams& opt_params) {
    AffineTransform cursor(transform);
    for (int step = 0; step < opt_params.desc_num_steps; step++) {
      double gradModule = cursor.gradStep(poly_real, poly_pred, opt_params.desc_lr,
                                          opt_params.reg_rate);

      Polygon tmp_poly(cursor.transform(poly_real));

      if (gradModule < 1e-5) {
        break;
      }
    }

    return cursor;
  }

  double normalize_theta(double theta) {
    return atan2(sin(theta), cos(theta));
  }
}

double Polygon::distance(const Point& p) const {
  double dist = std::numeric_limits<double>::max();
  for (uint32_t i = 0; i < points.size(); i++) {
    double new_dist = points[i].distance(p);
    if (new_dist < dist) {
      dist = new_dist;
    }
  }

  return dist;
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

double AffineTransform::gradStep(const Polygon& src, const Polygon& dst, double step, double lambda) {
  const Polygon& img = transform(src);

  double g_x = 0.;
  double g_y = 0.;
  double g_theta = 0.;
  double g_scale = 0.;
  double g_cx = 0.;
  double g_cy = 0.;

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
      const double alpha_x = a_i.x - c_x;
      const double alpha_y = a_i.y - c_y;
      const double sinT = sin(theta);
      const double cosT = cos(theta);

      const double s = scale;

      g_x += dist_x;
      g_y += dist_y;
      g_theta += dist_x * (alpha_x * s * (- sinT) + alpha_y * s * cosT)
                 + dist_y * (alpha_x * s * (- cosT) + alpha_y * s * (- sinT));
      g_scale += dist_x * (alpha_x * cosT + alpha_y * sinT)
                 + dist_y * (alpha_x * (- sinT) + alpha_y * cosT);
      g_cx += dist_x * (- s * cosT + 1.) + dist_y * (s * sinT + 1.);
      g_cy += dist_x * (- s * sinT + 1.) + dist_y * (- s * cosT + 1.);
    };
  }

  for (const Point& b_j: dst.get_points()) {
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
      const double alpha_x = a_i.x - c_x;
      const double alpha_y = a_i.y - c_y;
      const double sinT = sin(theta);
      const double cosT = cos(theta);

      const double s = scale;

      g_x += dist_x;
      g_y += dist_y;
      g_theta += dist_x * (alpha_x * s * (- sinT) + alpha_y * s * cosT)
                 + dist_y * (alpha_x * s * (- cosT) + alpha_y * s * (- sinT));
      g_scale += dist_x * (alpha_x * cosT + alpha_y * sinT)
                 + dist_y * (alpha_x * (- sinT) + alpha_y * cosT);
      g_cx += dist_x * (- s * cosT + 1.) + dist_y * (s * sinT + 1.);
      g_cy += dist_x * (- s * sinT + 1.) + dist_y * (- s * cosT + 1.);
    }
  }

  double num_points = src.size() + dst.size();
  g_x /= num_points;
  g_y /= num_points;
  g_theta /= num_points;
  g_scale /= num_points;
  g_cx /= num_points;
  g_cy /= num_points;

  const double g_module = sqrt(sqr(g_x) + sqr(g_y) + sqr(g_theta) + sqr(g_scale));
  shift_x -= (g_x + lambda * shift_x) * step;
  shift_y -= (g_y + lambda * shift_y) * step;
  theta -= (g_theta + lambda * theta) * step / 100;
  scale -= (g_scale - lambda * (1. - scale)) * step / 100;
  scale = std::min(1.5, std::max(0.5, scale));
//  c_x -= g_cx * step;
//  c_y -= g_cy * step;

  return g_module;
}

AffineResult find_affine(Polygon poly_real, Polygon poly_pred, const OptimizationParams& opt_params) {
  Point center1 = poly_real.center();

  poly_real.translate(-center1.x, -center1.y);
  poly_pred.translate(-center1.x, -center1.y);

  AffineTransform result = utils::grad_descent(poly_real, poly_pred,
                          {0, 0, 0, 1, 0, 0}, opt_params);

  Polygon tmp_real = result.transform(poly_real);
  double residual = utils::residual(tmp_real, poly_pred);
  result.theta = utils::normalize_theta(result.theta);
  return { result, residual };
}

void Polygon::calc_center() {
  _center = {0, 0};

  for (uint32_t i = 0; i < this->points.size(); i++) {
    const Point& point = this->points[i];
    _center.x += point.x;
    _center.y += point.y;
  }

  _center.x /= this->points.size();
  _center.y /= this->points.size();
}

Polygon::Polygon(const std::vector<Point>& points) {
  this->points = points;
  calc_center();
}

Polygon::Polygon(nlohmann::json& json) {
  points = std::vector<Point>(json.size() - 1);
  for (uint32_t i = 0; i < json.size() - 1; i++) {
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
        .def("set_reg_rate", &OptimizationParamsBuilder::set_reg_rate)
        .def("build", &OptimizationParamsBuilder::build);

    class_<OptimizationParams>("OptimizationParams",
        init<double, double, double, double, double, double, int, int, double, double>());

    class_<AffineTransform>("AffineTransform",
        init<double, double, double, double, double, double>())
        .add_property("shift_x" , &AffineTransform::shift_x)
        .add_property("shift_y" , &AffineTransform::shift_y)
        .add_property("theta"   , &AffineTransform::theta)
        .add_property("scale"   , &AffineTransform::scale);

    class_<AffineResult>("AffineResult")
        .add_property("transform", &AffineResult::transform)
        .add_property("residual", &AffineResult::residual);
}
#endif
