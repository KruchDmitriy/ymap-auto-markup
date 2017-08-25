#pragma once

#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>
#include <vector>
#include <cstdlib>

#include <boost/python/def.hpp>
#include <boost/python/module.hpp>
#include <boost/python/numpy.hpp>

namespace np = boost::python::numpy;

class oblivious_tree {
public:
    oblivious_tree(std::ifstream& model_file, double coeff, unsigned int depth = 6, bool log = false);
    double predict(double* x, size_t length);
    ~oblivious_tree();
private:
    unsigned int depth_;
    unsigned int* features;
    unsigned int* bins;
    double* conditions;
    double* values;
    double coeff_;
};

class ensemble {
public:
    ensemble(const char* model_file_name, unsigned int depth = 6, bool log = false);
    double predict(np::ndarray const& x);
private:
    size_t n_trees;
    std::vector<oblivious_tree*> trees;
};
