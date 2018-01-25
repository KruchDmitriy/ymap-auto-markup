#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <fstream>
#include <math.h>

#include "poly_match.hpp"
#include "3rd-party/json.hpp"
#include "3rd-party/utm/utm.h"
#include "LightGBM/boosting.h"
#include "LightGBM/prediction_early_stop.h"
#include "3rd-party/LightGBM/src/application/predictor.hpp"

enum Args {
    HELP,
    REAL,
    GEN,
    MODEL_FILE,
    MODEL_TYPE,
    MIN_SHIFT,
    MAX_SHIFT,
    MIN_THETA,
    MAX_THETA,
    MIN_SCALE,
    MAX_SCALE,
    GRID_STEP,
    DESC_NUM_STEPS,
    LEARN_RATE
};

static void small_help() {
    std::cout <<
    "usage: ./ymaprica [-h] --real REAL --gen GEN" << std::endl <<
    "              --model_file MODEL_FILE --model_type MODEL_TYPE" << std::endl <<
    "             [--min_shift MIN_SHIFT] [--max_shift MAX_SHIFT]" << std::endl <<
    "             [--min_theta MIN_THETA] [--max_theta MAX_THETA]" << std::endl <<
    "             [--min_scale MIN_SCALE] [--max_scale MAX_SCALE]" << std::endl <<
    "             [--grid_step GRID_STEP]" << std::endl <<
    "             [--desc_num_steps DESC_NUM_STEPS] [--learn_rate LEARN_RATE]" << std::endl <<
    "./ymaprica: error: the following arguments are required:"  << std::endl <<
    "              --real, --gen, --model_file, --model_type" << std::endl;
}

static void help() {
    std::cout <<
    "usage: ./ymaprica [-h] --real REAL --gen GEN" << std::endl <<
    "              --model_file MODEL_FILE --model_type MODEL_TYPE" << std::endl <<
    "             [--min_shift MIN_SHIFT] [--max_shift MAX_SHIFT]" << std::endl <<
    "             [--min_theta MIN_THETA] [--max_theta MAX_THETA]" << std::endl <<
    "             [--min_scale MIN_SCALE] [--max_scale MAX_SCALE]" << std::endl <<
    "             [--grid_step GRID_STEP]" << std::endl <<
    "             [--desc_num_steps DESC_NUM_STEPS] [--learn_rate LEARN_RATE]" << std::endl <<
    "arguments:" << std::endl <<
    "  -h, --help               show this help message and exit" << std::endl <<
    "  --real REAL              path to json file with list polygons [poly1, poly2, poly3]" << std::endl <<
    "  --gen GEN                path to json file with list generated polygons [gen_poly1," << std::endl <<
    "                           gen_poly2, gen_poly3]gen_poly1 will be compared with" << std::endl <<
    "                           poly1 etc" << std::endl <<
    "  --model_type MODEL_TYPE  linear or trees " << std::endl <<
    "  --model_file MODEL_FILE  path to model (from modeling.py, should be in data/linear.param)" << std::endl << std::endl <<
    "optimization parameters for grid search:" << std::endl <<
    "  --min_shift MIN_SHIFT" << std::endl <<
    "  --max_shift MAX_SHIFT" << std::endl <<
    "  --min_theta MIN_THETA" << std::endl <<
    "  --max_theta MAX_THETA" << std::endl <<
    "  --min_scale MIN_SCALE" << std::endl <<
    "  --max_scale MAX_SCALE" << std::endl <<
    "  --grid_step GRID_STEP" << std::endl << std::endl <<
    "optimization parameters for gradient descent:" << std::endl <<
    "  --desc_num_steps DESC_NUM_STEPS" << std::endl <<
    "  --learn_rate LEARN_RATE" << std::endl;
}

struct ParsedArgs {
    const char* file_real;
    const char* file_gen;
    const char* file_model;
    const char* model_type;
    OptimizationParams params;

    ParsedArgs(const char* file_real, const char* file_gen,
               const char* file_model, const char* model_type,
               OptimizationParams params)
    : file_real(file_real)
    , file_gen(file_gen)
    , file_model(file_model)
    , model_type(model_type)
    , params(params) {}
};

static ParsedArgs* parse_params(int argc, char** argv) {
    std::unordered_map<std::string, int> mapping = {
        {"-h", HELP}, {"--help", HELP},
        {"--real", REAL},
        {"--gen", GEN},
        {"--model_file", MODEL_FILE},
        {"--model_type", MODEL_TYPE},
        {"--min_shift", MIN_SHIFT},
        {"--max_shift", MAX_SHIFT},
        {"--min_theta", MIN_THETA},
        {"--max_theta", MAX_THETA},
        {"--min_scale", MIN_SCALE},
        {"--max_scale", MAX_SCALE},
        {"--grid_step", GRID_STEP},
        {"--desc_num_steps", DESC_NUM_STEPS},
        {"--learn_rate", LEARN_RATE}
    };

    OptimizationParamsBuilder builder;
    int required_completed = 0;
    char* file_real;
    char* file_gen;
    char* file_model;
    char* model_type;

    for (int i = 1; i < argc; i += 2) {
        switch (mapping[argv[i]]) {
            case HELP:
                help();
                return nullptr;
            case REAL:
                file_real = argv[i + 1];
                required_completed++;
                break;
            case GEN:
                file_gen = argv[i + 1];
                required_completed++;
                break;
            case MODEL_FILE:
                file_model = argv[i + 1];
                required_completed++;
                break;
            case MODEL_TYPE:
                model_type = argv[i + 1];
                required_completed++;
                break;
            case MIN_SHIFT: {
                double min_shift = atof(argv[i + 1]);
                builder.set_min_shift(min_shift);
                break;
            }
            case MAX_SHIFT: {
                double max_shift = atof(argv[i + 1]);
                builder.set_max_shift(max_shift);
                break;
            }
            case MIN_THETA: {
                double min_theta = atof(argv[i + 1]);
                builder.set_min_theta(min_theta);
                break;
            }
            case MAX_THETA: {
                double max_theta = atof(argv[i + 1]);
                builder.set_max_theta(max_theta);
                break;
            }
            case MIN_SCALE: {
                double min_scale = atof(argv[i + 1]);
                builder.set_min_scale(min_scale);
                break;
            }
            case MAX_SCALE: {
                double max_scale = atof(argv[i + 1]);
                builder.set_max_scale(max_scale);
                break;
            }
            case GRID_STEP: {
                int grid_step = atoi(argv[i + 1]);
                builder.set_grid_step(grid_step);
                break;
            }
            case DESC_NUM_STEPS: {
                int desc_num_steps = atoi(argv[i + 1]);
                builder.set_desc_num_steps(desc_num_steps);
                break;
            }
            case LEARN_RATE: {
                double learn_rate = atof(argv[i + 1]);
                builder.set_learn_rate(learn_rate);
                break;
            }
        }
    }

    if (required_completed < 4) {
        small_help();
        return nullptr;
    }

    return new ParsedArgs(file_real, file_gen, file_model, model_type, builder.build());
}

class Model {
public:
    virtual double predict(const AffineResult& result) = 0;
};

class LinearModel : public Model {
    std::vector<double> weights;
public:
    LinearModel(const char* model_file_path) {
        std::ifstream model_file(model_file_path);

        if (!model_file) {
            throw std::logic_error("File with model not found (should be in data/linear.param)");
        }

        double weight;
        while (model_file >> weight) {
            weights.push_back(weight);
        }

        assert(weights.size() == 6);
    }

    double predict(const AffineResult& result) override {
        AffineTransform transform = result.transform;
        double sum = weights[0] * abs(transform.shift_x)
                   + weights[1] * abs(transform.shift_y)
                   + weights[2] * abs(transform.theta)
                   + weights[3] * abs(1. - transform.scale)
                   + weights[4] * abs(result.residual)
                   + weights[5];
        return 1. / (1. + exp(-sum));
    }
};

class TreesModel : public Model {
    std::unique_ptr<LightGBM::Boosting> gbm;
    LightGBM::PredictionEarlyStopConfig pred_early_stop_config;
    LightGBM::PredictionEarlyStopInstance early_stop;
    double output;
public:
    TreesModel(const char* model_file) {
        using namespace LightGBM;
        gbm = std::unique_ptr<Boosting>(
            Boosting::CreateBoosting("gbdt", model_file));
        early_stop = CreatePredictionEarlyStopInstance(
            "binary", pred_early_stop_config);
    }

    double predict(const AffineResult& result) override {
        auto transform = result.transform;
        double features[] = {
            abs(transform.shift_x),
            abs(transform.shift_y),
            abs(transform.theta),
            abs(1. - transform.scale),
            abs(result.residual)
        };
        gbm->Predict(features, &output, &early_stop);
        return output;
    }
};

double calc_metric(const Polygon& poly_real, const Polygon& poly_gen,
                   const OptimizationParams& params, Model* model) {
    AffineResult result = find_affine(poly_real, poly_gen, params);
    return model->predict(result);
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

int main(int argc, char** argv) {
    using json = nlohmann::json;

    ParsedArgs* args = parse_params(argc, argv);
    if (args == nullptr) {
        return 1;
    }

    std::ifstream file_real(args->file_real);
    std::ifstream file_gen(args->file_gen);

    if (!file_real) {
        throw std::logic_error(std::string("File ") + args->file_real + " not found");
    }

    if (!file_gen) {
        throw std::logic_error(std::string("File ") + args->file_gen + " not found");
    }

    json real, gen;
    file_real >> real;
    file_gen >> gen;

    if (real.size() != gen.size()) {
        std::cout << "Files length differ" << std::endl;
        return 1;
    }

    Model* model;
    std::string model_type = args->model_type;
    if (model_type == "linear") {
        model = new LinearModel(args->file_model);
    } else if (model_type == "trees") {
        model = new TreesModel(args->file_model);
    } else {
        throw std::logic_error("Unknown model type " + model_type);
    }

    for (uint32_t i = 0; i < real.size(); i++) {
        Polygon real_poly = points_to_poly(real[i]);
        Polygon gen_poly = points_to_poly(gen[i]);
        std::cout << std::setprecision(6)
            << calc_metric(real_poly, gen_poly, args->params, model) << std::endl;
    }

    delete args;

    return 0;
}
