#include "tree.h"

namespace bp = boost::python;
namespace np = boost::python::numpy;


oblivious_tree::oblivious_tree(std::ifstream& model_file, double coeff, unsigned int depth, bool log)
: depth_(depth)
, coeff_(coeff)
{
    features = new unsigned int[depth];
    bins = new unsigned int[depth];
    conditions = new double[depth];
    const unsigned int pow_depth = 1 << depth;
    values = new double[pow_depth];

    for (size_t i = 0; i < pow_depth; i++) {
        values[i] = 0.;
    }

    std::string line;

    for (unsigned int i = 0; i < depth; i++) {
        std::getline(model_file, line);
        std::stringstream ss(line);

        std::string value;
        ss >> value;
        ss >> value;

        features[i] = std::atoi(value.substr(0, value.size() - 1).c_str());

        ss >> value;
        ss >> value;

        bins[i] = std::atoi(value.substr(0, value.size() - 1).c_str());

        ss >> value;
        ss >> value;

        conditions[i] = std::atof(value.c_str());

        if (log) {
            std::cout << i << "th level" << std::endl;
            std::cout << line << std::endl;
            std::cout << features[i] << " " << bins[i] << " " << conditions[i] << std::endl;
        }
    }

    std::getline(model_file, line);

    std::stringstream ss(line);
    std::string value;
    while (ss >> value) {
        size_t first_pos = value.find_first_of(":");
        size_t last_pos = value.find_last_of(":");

        char *p_end;
        std::string str_idx = value.substr(0, first_pos);
        unsigned int idx = std::strtol(str_idx.c_str(), &p_end, 2);
        double mean = std::atof(value.substr(first_pos + 1, last_pos).c_str());

        if (log) {
            std::cout << value << std::endl;
            std::cout << idx << " " << mean << std::endl;
        }

        values[idx] = mean;
    }

    if (log) {
        std::cout << "tree was built" << std::endl;
    }
}

double oblivious_tree::predict(double* x, size_t length) {
    unsigned int idx = 0;

    for (unsigned int i = 0; i < depth_; i++) {
        unsigned int val = (x[features[i]] >= conditions[i]) ? 1 : 0;
        idx = val + idx * 2;
    }

    return values[idx] * coeff_;
}

oblivious_tree::~oblivious_tree() {
    delete[] features;
    delete[] conditions;
    delete[] values;
}

ensemble::ensemble(const char* model_file_name, unsigned int depth, bool log) {
    if (log) {
        std::cout << "loading model " << model_file_name << std::endl;
    }

    std::ifstream model_file(model_file_name);

    std::string num_trees;
    std::getline(model_file, num_trees);
    std::getline(model_file, num_trees);

    n_trees = std::atoi(num_trees.c_str());

    if (log) {
        std::cout << n_trees << std::endl;
    }

    std::string line;
    for (size_t i = 0; i < n_trees; i++) {
        std::getline(model_file, line);

        if (line.find("ObliviousTree") != std::string::npos) {
            size_t pos = line.find_last_of(" ");
            double coeff = std::atof(line.substr(pos + 1, line.size()).c_str());

            if (log) {
                std::cout << coeff << std::endl;
            }
            trees.push_back(new oblivious_tree(model_file, coeff, depth, log));
        }
    }
}

double ensemble::predict(np::ndarray const& x) {
    double* x_ = reinterpret_cast<double*>(x.get_data());
    size_t length = x.shape(0);

    double result = 0.;
    for (size_t i = 0; i < trees.size(); i++) {
        result += trees[i]->predict(x_, length);
    }

    return result;
}

BOOST_PYTHON_MODULE(tree) {
    using namespace boost::python;

    np::initialize();

    class_<ensemble>("Ensemble", init<const char*, int, bool>())
        .def("predict", &ensemble::predict);
}
