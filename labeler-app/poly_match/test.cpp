#define CATCH_CONFIG_MAIN

#include "3rd-party/catch.h"
#include "3rd-party/json.hpp"
#include "poly_match.hpp"
#include "3rd-party/utm/utm.h"


using Catch::Matchers::WithinAbs;

class TestData {
public:
    static const OptimizationParams opt_params;
    static const Polygon simple_poly;
    static const Polygon pi_poly;
    static const Polygon z_poly;
};

const OptimizationParams TestData::opt_params = OptimizationParamsBuilder().build();
const Polygon TestData::simple_poly = Polygon({{1, 1}, {1, 4}, {4, 4}, {4, 1}});
const Polygon TestData::pi_poly = Polygon({{ 6179662.5325047, 588730.9532652 },
                                          { 6179615.63754466, 588777.51369495},
                                          { 6179610.52244683, 588772.35574043},
                                          { 6179606.65420892, 588776.20252638},
                                          { 6179560.61964376, 588729.83303407},
                                          { 6179564.63268182, 588725.85697921},
                                          { 6179560.63536086, 588721.83376113},
                                          { 6179607.39749536, 588675.41297654},
                                          { 6179662.5325047 , 588730.9532652 }});
const Polygon TestData::z_poly = Polygon({{ 6181614.98270132,   589459.24887207 },
                                          { 6181622.27635998,   589466.56797726 },
                                          { 6181585.77278982,   589502.87722498 },
                                          { 6181657.3713091 ,   589574.86309694 },
                                          { 6181589.13913586,   589642.73501421 },
                                          { 6181545.85643095,   589599.22219838 },
                                          { 6181553.40229229,   589591.71313077 },
                                          { 6181589.04687789,   589627.54604101 },
                                          { 6181641.71348966,   589575.16471437 },
                                          { 6181570.45840057,   589503.52926773 },
                                          { 6181614.98270132,   589459.24887207 }});


TEST_CASE("test find identity", "[find_affine]") {
  AffineResult result = find_affine(TestData::simple_poly, TestData::simple_poly, TestData::opt_params);
  REQUIRE(result.residual <= 1e-6);
  REQUIRE(result.transform.regularization() <= 1e-6);
}

void test_center_rotate(const Polygon &poly) {
  const int n_rotations = 100;
  double theta = -0.3;
  const double theta_step = (0.3 - theta) / n_rotations;

  for (int i = 0; i < n_rotations; i++) {
    theta += theta_step;
    Polygon gen = poly;
    gen.translate(-gen.center().x, -gen.center().y);
    gen.rotate(theta);
    gen.translate(+gen.center().x, +gen.center().y);

    AffineResult result = find_affine(poly, gen, TestData::opt_params);
    REQUIRE_THAT(result.transform.theta, WithinAbs(theta, 1e-3));
    REQUIRE(result.residual <= 1e-4);
  }
}

void test_center_scale(const Polygon& poly) {
  const int n_scales = 100;
  double scale = 0.7;
  const double scale_step = (1.3 - scale) / n_scales;

  for (int i = 0; i < n_scales; i++) {
    scale += scale_step;
    Polygon gen = poly;
    gen.translate(-gen.center().x, -gen.center().y);
    gen.scale(scale);
    gen.translate(gen.center().x, gen.center().y);

    AffineResult result = find_affine(poly, gen, TestData::opt_params);
    REQUIRE_THAT(result.transform.scale, WithinAbs(scale, 1e-3));
    REQUIRE(result.residual <= 1e-4);
  }
}

double rng(double min, double max) {
  return std::rand() / RAND_MAX * (max - min) + min;
}

void test_random_trans(const Polygon& poly) {
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  double theta;
  double scale;

  for (int i = 0; i < 100; i++) {
    theta = rng(-0.3, 0.3);
    scale = rng(0.7, 1.3);

    Polygon gen = poly;
    gen.translate(-gen.center().x, -gen.center().y);
    gen.scale(scale);
    gen.rotate(theta);
    gen.translate(gen.center().x, gen.center().y);

    AffineResult result = find_affine(poly, gen, TestData::opt_params);
    REQUIRE_THAT(result.transform.theta, WithinAbs(theta, 1e-3));
    REQUIRE_THAT(result.transform.scale, WithinAbs(scale, 1e-3));
    REQUIRE(result.residual <= 1e-4);
  }
}

TEST_CASE("test find rotate simple", "[find_affine]") {
  test_center_rotate(TestData::simple_poly);
}

TEST_CASE("test find scale simple", "[find_affine]") {
  test_center_scale(TestData::simple_poly);
}

TEST_CASE("test find rotate z poly", "[find_affine]") {
  test_center_rotate(TestData::z_poly);
}

TEST_CASE("test find scale z poly", "[find_affine]") {
  test_center_scale(TestData::z_poly);
}

TEST_CASE("test find random z poly", "[find_affine]") {
  test_random_trans(TestData::z_poly);
}

TEST_CASE("test find rotate pi poly", "[find_affine]") {
  test_center_rotate(TestData::pi_poly);
}

TEST_CASE("test find scale pi poly", "[find_affine]") {
  test_center_scale(TestData::pi_poly);
}

void convert_coords_utm(double lat, double lon, double* east, double* north) {
  long zone;
  char hemisphere;

  lat *= M_PI / 180.;
  lon *= M_PI / 180.;

  long error = Convert_Geodetic_To_UTM(lat, lon, &zone, &hemisphere, east, north);
  if (error != UTM_NO_ERROR) {
    std::cout << error << std::endl;
    std::cout << *east << " " << *north << std::endl;
    throw std::logic_error(std::string("Cannot convert latlon coordinates ")
                           + std::to_string(lat) + " " + std::to_string(lon) + " to utm");
  }
}

Polygon points_to_poly(nlohmann::json& points) {
  for (uint32_t i = 0; i < points.size(); i++) {
    double east, north;
    convert_coords_utm(points[i][1], points[i][0], &east, &north);
    points[i][0] = north;
    points[i][1] = east;
  }

  return Polygon(points);
}

TEST_CASE("test real data") {
  using json = nlohmann::json;
  std::ifstream file_real("../json_examples/real.json");
  std::ifstream file_gen("../json_examples/gen.json");

  json real, gen;
  file_real >> real;
  file_gen >> gen;

  for (uint32_t i = 0; i < real.size(); i++) {
    if (i == 762) {
      continue;
    }
    Polygon real_poly = points_to_poly(real[i]);
    Polygon gen_poly = points_to_poly(gen[i]);
    std::cout << i << std::endl;
    AffineResult result = find_affine(real_poly, gen_poly, TestData::opt_params);
    CHECK_THAT(result.residual, WithinAbs(0., 0.6));
  }
}