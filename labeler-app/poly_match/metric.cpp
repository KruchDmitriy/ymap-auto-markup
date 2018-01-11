#include <iostream>
#include <unordered_map>
#include <fstream>
#include "poly_match.hpp"
#include "3rd-party/json.hpp"

enum Args {
    HELP,
    REAL,
    GEN,
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
    "usage: ./metric [-h] --real REAL --gen GEN [--min_shift MIN_SHIFT]" << std::endl <<
    "             [--max_shift MAX_SHIFT] [--min_theta MIN_THETA]" << std::endl <<
    "             [--max_theta MAX_THETA] [--min_scale MIN_SCALE]" << std::endl <<
    "             [--max_scale MAX_SCALE] [--grid_step GRID_STEP]" << std::endl <<
    "             [--desc_num_steps DESC_NUM_STEPS] [--learn_rate LEARN_RATE]" << std::endl <<
    "./metric: error: the following arguments are required: --real, --gen" << std::endl;
}

static void help() {
    std::cout <<
    "usage: metric.py [-h] --real REAL --gen GEN [--min_shift MIN_SHIFT]" << std::endl <<
    "             [--max_shift MAX_SHIFT] [--min_theta MIN_THETA]" << std::endl <<
    "             [--max_theta MAX_THETA] [--min_scale MIN_SCALE]" << std::endl <<
    "             [--max_scale MAX_SCALE] [--grid_step GRID_STEP]" << std::endl <<
    "             [--desc_num_steps DESC_NUM_STEPS] [--learn_rate LEARN_RATE]" << std::endl << std::endl <<
    "arguments:" << std::endl <<
    "  -h, --help            show this help message and exit" << std::endl <<
    "  --real REAL           path to json file with list polygons [poly1, poly2, poly3]" << std::endl <<
    "  --gen GEN             path to json file with list generated polygons [gen_poly1," << std::endl <<
    "                        gen_poly2, gen_poly3]gen_poly1 will be compared with" << std::endl <<
    "                        poly1 etc" << std::endl << std::endl <<
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
    char* file_real;
    char* file_gen;
    OptimizationParams params;

    ParsedArgs(char* file_real, char* file_gen, OptimizationParams params)
    : file_real(file_real)
    , file_gen(file_gen)
    , params(params) {}
};

static ParsedArgs* parse_params(int argc, char** argv) {
    std::unordered_map<std::string, int> mapping = {
        {"-h", HELP}, {"--help", HELP},
        {"--real", REAL},
        {"--gen", GEN},
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

    if (required_completed < 2) {
        small_help();
        return nullptr;
    }

    return new ParsedArgs(file_real, file_gen, builder.build());
}

class LinearModel {
    std::vector<double> weights;
public:
    LinearModel() {
        std::ifstream model_file("../data/linear.params");
        double weight;
        while (model_file >> weight) {
            weights.push_back(weight);
        }

        assert(weights.size() == 6);
    }

    double predict(const AffineResult& result) const {
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


double calc_metric(const Polygon& poly_real, const Polygon& poly_gen,
                   const OptimizationParams& params, const LinearModel& model) {
    AffineResult result = find_affine(poly_real, poly_gen, params);
    return model.predict(result);
}

int main(int argc, char** argv) {
    using json = nlohmann::json;

    ParsedArgs* args = parse_params(argc, argv);
    if (args == nullptr) {
        return 1;
    }

    std::ifstream file_real(args->file_real);
    std::ifstream file_gen(args->file_gen);
    json real, gen;
    file_real >> real;
    file_gen >> gen;

    if (real.size() != gen.size()) {
        std::cout << "Files length differ" << std::endl;
        return 1;
    }

    LinearModel model;

    for (uint32_t i = 0; i < real.size(); i++) {
        std::cout << calc_metric(Polygon(real[i]), Polygon(gen[i]),
                                 args->params, model) << std::endl;
    }

    delete args;
}
