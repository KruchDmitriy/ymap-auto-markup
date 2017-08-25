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

double target_function(double alpha, double beta, double tau_0, double tau_1, const cv::Mat& image) {
    double result = 0.;

    double ax = cos(alpha), ay = sin(alpha);
    double bx = cos(alpha + CV_PI/2), by = sin(alpha + CV_PI/2);
    double basisX = beta * bx;
    double basisY = beta * by;

    double sum = 0;
    double sum2 = 0;
    double totalW = 0;
    for (int i = 0; i < image.rows; i++) {
        for (int j = i + 1; j < image.cols; j++) {
            double x = i - basisX;
            double y = j - basisY;
            // transfering to new basis
            const double projA = x * ax + y * ay;
            const double projB = x * bx + y * by;
            if (fabs(projB) < 1 || (projA - tau_0) * (projA - tau_1) > 0)
                continue;
            double signum = projB >= 0 ? 1.0 : -1.0;

            const double weight = -projB * projB;

            const double mirrorX = projA * ax - projB * bx + basisX;
            const double mirrorY = projA * ay - projB * by + basisY;

            int mirrorJ = (int)round(mirrorY);
            int mirrorI = (int)round(mirrorX);

            if (mirrorI < 0 || mirrorJ < 0 || mirrorI >= image.rows || mirrorJ >= image.cols)
                continue;

            double proBabOrig = image.at<double>(i, j);
            double proBabMirror = image.at<double>(mirrorI, mirrorJ);
            double logOdds = log(proBabOrig) - log(proBabMirror);
            static double maxOdds = 0;
            if (fabs(logOdds) > maxOdds) {
                maxOdds = fabs(logOdds);
                std::cout << logOdds << std::endl;
            }
            double w = exp(weight / 2);
            sum += w * signum * logOdds;
            sum2 += w * logOdds * logOdds;
            totalW += w;
            result += signum * w * logOdds;
        }
    }

    double D = totalW > 0 ? (sum2 - sum * sum / totalW) : 0;
    if (D > 0) {
//        if (D > 1)
//            std::cout << "Hello";
        return result - D;
//        if (result > 9)
//            target_function(alpha, beta, tau_0, tau_1, image);
//        return result;
    }
    return 0;
}

void optimize(const cv::Mat& img, const cv::Mat& img_norm) {
    const size_t num_instances = 360;

    int side = std::max(img_norm.cols, img_norm.rows);
    size_t counter = 0;

    for (size_t i = 0; i < num_instances; i++) {
        for (size_t j = 0; j < side * 2; j++) {
            for (size_t k = 0; k < side; k++) {
                for (size_t l = k; l < side; l++) {
                    double alpha = 2. * M_PI * i / num_instances;
                    double beta = j - side;
                    double t_0 = k;
                    double t_1 = l;
                    double value = target_function(alpha, beta, t_0, t_1, img_norm);

                    if (value > 5) {
                        double ax = cos(alpha), ay = sin(alpha);
                        double bx = cos(alpha + CV_PI / 2), by = sin(alpha + CV_PI / 2);
                        
                        int x_0 = (int) round(t_0 * ax + beta * bx);
                        int y_0 = (int) round(t_0 * ay + beta * by);

                        int x_1 = (int) round(t_1 * ax + beta * bx);
                        int y_1 = (int) round(t_1 * ay + beta * by);

                        std::cout << "score " << value << " ";
                        std::cout << x_0 << " " << y_0 << " " << x_1 << " " << y_1 << std::endl;
                        std::cout << "alpha " << alpha << " beta " << beta << " tau0 "
                                  << t_0 << " tau1 " << t_1 << std::endl;

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

    img = imread("../results/hog_10K_small.png");
//    resize(img, img, cv::Size(100, 100));
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
