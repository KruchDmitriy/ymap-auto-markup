#include "circle_desc.hpp"

#include <iomanip>

#include <cstring>
#include <cmath>

namespace p = boost::python;
namespace np = boost::python::numpy;

circle_descriptor::circle_descriptor(double radius, uint32_t num_circles, bool log)
: radius(radius)
, num_circles(num_circles)
, log(log) {
    create_rings();
}

void circle_descriptor::create_rings() {
    int r = (int) round(radius);
    double stride = radius / num_circles;

    for (int x = -r; x <= r; x++) {
        for (int y = -r; y <= r; y++) {
            uint32_t ring = (uint32_t) round(sqrt(x * x + y * y) / stride);

            if (ring >= num_circles) {
                ring_idx.push_back(-1);
                continue;
            }

            ring_idx.push_back(ring);
        }
    }

    if (log) {
        const int window_width = 2 * r + 1;
        for (uint32_t i = 0; i < ring_idx.size(); i++) {
            std::cout << std::setw(3) << ring_idx[i] << " ";

            if (((int) i % window_width) == (2 * r))
                std::cout << std::endl;
        }
    }
}

np::ndarray circle_descriptor::compute(np::ndarray const& image, int i, int j) {
    int r = (int) round(radius);
    int height = image.shape(0);
    int width = image.shape(1);

    double mean_b[num_circles];
    double mean_g[num_circles];
    double mean_r[num_circles];

    double var_b[num_circles];
    double var_g[num_circles];
    double var_r[num_circles];

    int n_samples[num_circles];

    std::memset(mean_b, 0, sizeof(mean_b));
    std::memset(mean_g, 0, sizeof(mean_g));
    std::memset(mean_r, 0, sizeof(mean_r));

    std::memset(var_b, 0, sizeof(var_b));
    std::memset(var_g, 0, sizeof(var_g));
    std::memset(var_r, 0, sizeof(var_r));

    std::memset(n_samples, 0, sizeof(n_samples));

    uint8_t *data = reinterpret_cast<uint8_t*>(image.get_data());
    const int window_width = 2 * r + 1;

    for (int x = -r; x <= r; x++) {
        for (int y = -r; y <= r; y++) {
            if (i + x < 0 || i + x >= height || j + y < 0 || j + y >= height)
                continue;

            int index = (x + r) * window_width + (y + r);
            int image_index = (i + x) * width + (j + y);

            if (ring_idx[index] == -1)
                continue;

            n_samples[ring_idx[index]]++;

            mean_b[ring_idx[index]] += data[image_index * 3];
            mean_g[ring_idx[index]] += data[image_index * 3 + 1];
            mean_r[ring_idx[index]] += data[image_index * 3 + 2];

            var_b[ring_idx[index]] += data[image_index * 3] * data[image_index * 3];
            var_g[ring_idx[index]] += data[image_index * 3 + 1] * data[image_index * 3 + 1];
            var_r[ring_idx[index]] += data[image_index * 3 + 2] * data[image_index * 3 + 2];
        }
    }

    for (uint32_t i = 0; i < num_circles; i++) {
        if (n_samples[i] == 1) {
            var_b[i] = 0.;
            var_g[i] = 0.;
            var_r[i] = 0.;
        } else {
            var_b[i] = (var_b[i] - mean_b[i] * mean_b[i] / n_samples[i]) / (n_samples[i] - 1);
            var_g[i] = (var_g[i] - mean_g[i] * mean_g[i] / n_samples[i]) / (n_samples[i] - 1);
            var_r[i] = (var_r[i] - mean_r[i] * mean_r[i] / n_samples[i]) / (n_samples[i] - 1);
        }

        mean_b[i] /= n_samples[i];
        mean_g[i] /= n_samples[i];
        mean_r[i] /= n_samples[i];

        if (log) {
            std::cout << "mean " << mean_b[i] << " " << mean_g[i] << " " << mean_r[i] << std::endl;
        }
    }

    std::vector<double> result(num_circles * 6);
    for (uint32_t i = 0; i < num_circles; i++) {
        result[6 * i]     = mean_b[i];
        result[6 * i + 1] = mean_g[i];
        result[6 * i + 2] = mean_r[i];
        result[6 * i + 3] = var_b[i];
        result[6 * i + 4] = var_g[i];
        result[6 * i + 5] = var_r[i];
    }

    Py_intptr_t shape[1] = { (Py_intptr_t) result.size() };
    np::ndarray res = np::zeros(1, shape, np::dtype::get_builtin<double>());
    std::copy(result.begin(), result.end(), reinterpret_cast<double*>(res.get_data()));
    return res;
}

BOOST_PYTHON_MODULE(circle_desc) {
    using namespace boost::python;

    np::initialize();

    class_<circle_descriptor>("CircleDescriptor", init<double, uint32_t, bool>())
        .def("compute", &circle_descriptor::compute);
}
