#include <iostream>
#include <tuple>
#include <iomanip>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "vec.h"

cv::Mat img;
cv::Mat img_norm;

const double MIN_VAL = 1e-5;
const double MIN_LOG_VAL = log(MIN_VAL);
const double lambda = 0.1;
const double line_width = 10.;

std::tuple<int, int, int, int> calc_coords(double alpha, double beta, double tau_0, double tau_1) {
    double cs = cos(alpha);
    double sn = sin(alpha);

    int x_0 = tau_0 * cs + beta * sn;
    int y_0 = tau_0 * sn - beta * cs;

    int x_1 = tau_1 * cs + beta * sn;
    int y_1 = tau_1 * sn - beta * cs;

    return std::make_tuple(x_0, y_0, x_1, y_1);
}

double dot(const vec& v1, const vec& v2) {
    return v1.x * v2.x + v1.y * v2.y;
}

double target_function(double alpha, double beta, double tau_0, double tau_1) {
    double result = 0.;
    bool was_set = false;

    vec a { cos(alpha), sin(alpha) };
    vec b { sin(alpha), -cos(alpha) };

    for (int i = 0; i < img_norm.rows; i++) {
        for (int j = 0; j < img_norm.cols; j++) {
            vec x { (double)i, (double)j };


            if ((dot(a, x) - tau_0) < 0 || (tau_1 - dot(a, x) < 0) ||
                abs(dot(b, x) - beta) > line_width)
            {
                continue;
            }

            double p_x = log(img_norm.at<double>(i, j));

            vec inv_x = x - 2. * (dot(b, x) - beta) * b;
            int inv_i = (int)(inv_x.x + 0.5);
            int inv_j = (int)(inv_x.y + 0.5);

            double p_inv_x = (inv_i < img_norm.rows) && (inv_i >= 0) &&
                            (inv_j < img_norm.cols) && (inv_j >= 0) ?
                            log(img_norm.at<double>(inv_i, inv_j)) : MIN_LOG_VAL;

            double value = /*-(dot(b, x) - beta) * (dot(b, x) - beta) */ (p_x - p_inv_x)
                            + lambda * abs(tau_0 - tau_1);

            if (std::isnan(value)) {
                continue;
            }

            if (!was_set) {
                result = value;
                was_set = true;
                continue;
            }

            result += value;
        }
    }

    if (!was_set) {
        return -DBL_MAX;
    }

    return result;
}

void optimize() {
    const size_t num_instances = 100;

    size_t side = std::max(img_norm.cols, img_norm.rows);
    double max_beta_value = sqrt(2.) / 2. * side;

    double alphas[num_instances];
    double betas[side];

    double tau_0[side];
    double tau_1[side];

    for (size_t i = 0; i < num_instances; i++) {
        alphas[i] = 2. * M_PI * i / num_instances;
    }

    for (size_t i = 0; i < side; i++) {
        betas[i] = ((double) i / side) * 2. * max_beta_value - max_beta_value;
        tau_0[i] = ((double) i / side) * 2. * max_beta_value - max_beta_value;
        tau_1[i] = ((double) i / side) * 2. * max_beta_value - max_beta_value;
    }

    double max_value = -DBL_MAX;

    size_t counter = 0;

    for (size_t i = 0; i < num_instances; i++) {
        for (size_t j = 0; j < side; j++) {
            for (size_t k = 0; k < side; k++) {
                for (size_t l = 0; l < side; l++) {
                    auto tpl = calc_coords(alphas[i], betas[j], tau_0[k], tau_1[l]);
                    int x_0 = std::get<0>(tpl);
                    int y_0 = std::get<1>(tpl);
                    int x_1 = std::get<2>(tpl);
                    int y_1 = std::get<3>(tpl);

                    if (x_0 < img.rows / 4 || x_0 > 3 * img.rows / 4
                     || y_0 < img.cols / 4 || y_0 > 3 * img.cols / 4
                     || x_1 < img.rows / 4 || x_1 > 3 * img.rows / 4
                     || y_1 < img.cols / 4 || y_1 > 3 * img.cols / 4)
                    {
                        continue;
                    }

                    double value = target_function(alphas[i], betas[j], tau_0[k], tau_1[l]);

                    if (value > max_value) {
                        max_value = value;

                        std::cout << "score " << value << " ";
                        std::cout << x_0 << " " << y_0 << " " << x_1 << " " << y_1 << std::endl;
                        std::cout << "alpha " << alphas[i] << " beta " << betas[j] << " tau0 "
                            << tau_0[k] << " tau1 " << tau_1[l] << std::endl;

                        cv::Mat copy_img = img.clone();
                        cv::line(copy_img, cv::Point(y_0, x_0), cv::Point(y_1, x_1), cv::Scalar(255, 0, 0));
                        // cv::line(copy_img, cv::Point(0, 10), cv::Point(20, 30), cv::Scalar(255, 0, 0));

                        std::ostringstream ss;
                        ss << "line_" << std::setw(5) << std::setfill('0') << counter << ".png";
                        counter++;
                        std::string file_name = ss.str();

                        cv::imwrite(file_name, copy_img);
                    }
                }
            }
        }
    }
}


int main() {
    using namespace std;
    using namespace cv;

    img = imread("../../hog_10K_small.png");

    // for (int i = 0; i < img.rows; i++) {
    //     for (int j = 0; j < img.cols; j++) {
    //         uint8_t value = (i <= j) ? 0 : 255;
    //         img.at<Vec3b>(i, j) = Vec3b(value, value, value);
    //         // cout << (int)img.at<uint8_t>(i, j) << " ";
    //     }
    //     // cout << endl;
    // }

    normalize(img, img_norm, MIN_VAL, 1., NORM_MINMAX, CV_64F);

    // for (int i = 0; i < img.rows; i++) {
    //     for (int j = 0; j < img.cols; j++) {
    //         cout << img_norm.at<double>(i, j) << " ";
    //     }

    //     cout << endl;
    // }


    optimize();

    return 0;
}
