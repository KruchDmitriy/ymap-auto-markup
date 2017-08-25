#include <math.h>
#include <iostream>
#include <tuple>
#include <iomanip>
#include <sstream>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

const double MIN_VAL = 1e-5;

using namespace boost::numeric::ublas;

typedef vector<double, std::vector<double>> vec;

std::tuple<int, int, int, int> calc_coords(double alpha, double beta, double tau_0, double tau_1) {
    double cs = cos(alpha);
    double sn = sin(alpha);

    int x_0 = tau_0 * cs + beta * sn;
    int y_0 = tau_0 * sn - beta * cs;

    int x_1 = tau_1 * cs + beta * sn;
    int y_1 = tau_1 * sn - beta * cs;

    return std::make_tuple(x_0, y_0, x_1, y_1);
}

double target_function(double alpha, double beta, double tau_0, double tau_1, const cv::Mat& image) {
    double result = 0.;

    double ax = cos(alpha), ay = sin(alpha);
    double bx = cos(alpha + CV_PI/2), by = sin(alpha + CV_PI/2);
    double basisX = beta * bx;
    double basisY = beta * by;

    for (int i = image.rows / 3; i < image.rows * 2/ 3; i++) {
        for (int j = image.cols / 3; j < image.cols * 2 / 3; j++) {
            double x = i - basisX;
            double y = j - basisY;
            // transfering to new basis
            const double projA = x * ax + y * ay;
            const double projB = x * bx + y * by;
            double signum = projB >= 0 ? 1.0 : -1.0;

            if ((projA - tau_0) * (projA - tau_1) > 0)
                continue;

            const double weight = -projB * projB;

            const double mirrorX = projA * ax - projB * bx + basisX;
            const double mirrorY = projA * ay - projB * by + basisY;

            int mirrorJ = (int)round(mirrorY);
            int mirrorI = (int)round(mirrorX);

            if (mirrorI < 0 || mirrorJ < 0 || mirrorI >= image.rows || mirrorJ >= image.cols)
                return 0;

            double proBabOrig = image.at<double>(i, j);
            double proBabMirror = image.at<double>(mirrorI, mirrorJ);
            double logOdds = log(proBabOrig) - log(proBabMirror);
            result += signum * exp(weight * 2) * logOdds;
        }
    }

    return result;
}

void optimize(const cv::Mat& img, const cv::Mat& img_norm) {
    const size_t num_instances = 100;

    int side = std::max(img_norm.cols, img_norm.rows);

    double alphas[num_instances];
    double betas[2 * side];

    double tau_0[2 * side];
    double tau_1[2 * side];

    for (size_t i = 0; i < num_instances; i++) {
        alphas[i] = 2. * M_PI * i / num_instances;
    }

    for (int i = -side; i < side; i++) {
        betas[i + side] = i;
        tau_0[i + side] = i;
        tau_1[i + side] = i;
    }

    double max_value = -DBL_MAX;

    size_t counter = 0;

    for (size_t i = 0; i < num_instances; i++) {
        for (size_t j = 0; j < side * 2; j++) {
            for (size_t k = 0; k < side * 2; k++) {
                for (size_t l = 0; l < side * 2; l++) {
                    auto tpl = calc_coords(alphas[i], betas[j], tau_0[k], tau_1[l]);
                    int x_0 = std::get<0>(tpl);
                    int y_0 = std::get<1>(tpl);
                    int x_1 = std::get<2>(tpl);
                    int y_1 = std::get<3>(tpl);

                    if (x_0 < img.rows / 3 || x_0 > 2 * img.rows / 3
                     || y_0 < img.cols / 3 || y_0 > 2 * img.cols / 3
                     || x_1 < img.rows / 3 || x_1 > 2 * img.rows / 3
                     || y_1 < img.cols / 3 || y_1 > 2 * img.cols / 3)
                    {
                        continue;
                    }

                    double value = target_function(alphas[i], betas[j], tau_0[k], tau_1[l], img_norm);

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

    cv::Mat img;
    cv::Mat img_gray;
    cv::Mat img_norm;

    img = imread("../results/hog_10K.png");
    resize(img, img, cv::Size(100, 100));
    cvtColor(img, img_gray, COLOR_BGR2GRAY);

//    for (int i = 0; i < img.rows; i++) {
//        for (int j = 0; j < img.cols; j++) {
//            uint8_t value = (i <= j) ? 0 : 255;
//            img.at<uint8_t>(i, j) = value;
//            // cout << (int)img.at<uint8_t>(i, j) << " ";
//        }
//        // cout << endl;
//    }

    normalize(img_gray, img_norm, MIN_VAL, 1., NORM_MINMAX, CV_64F);

//    cout << img_norm.channels() << endl
//         << "\\Pi/2: " << target_function(CV_PI / 4, 0, 20, 40, img_norm)
//        << "-\\Pi/2: " << target_function(CV_PI+CV_PI / 4, 0, -20, -40, img_norm);

    // for (int i = 0; i < img.rows; i++) {
    //     for (int j = 0; j < img.cols; j++) {
    //         cout << img_norm.at<double>(i, j) << " ";
    //     }

    //     cout << endl;
    // }


    optimize(img, img_norm);

    return 0;
}
