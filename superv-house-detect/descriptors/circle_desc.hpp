#pragma once

#include <iostream>
#include <vector>

#include <boost/python/def.hpp>
#include <boost/python/module.hpp>
#include <boost/python/numpy.hpp>

namespace np = boost::python::numpy;

class circle_descriptor {
public:
    circle_descriptor(double radius, uint32_t num_circles, bool log = false);
    np::ndarray compute(np::ndarray const& image, int i, int j);
private:
    void create_rings();

    std::vector<int> ring_idx;
    double radius;
    uint32_t num_circles;
    bool log;
};
