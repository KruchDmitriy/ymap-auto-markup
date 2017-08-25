#include <iostream>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;
using namespace std;


vector<Mat> calc_hist(Mat& img) {
    vector<Mat> bgr_planes;
    split(img, bgr_planes);

    // Establish the number of bins
    int histSize = 256;

    // Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat b_hist, g_hist, r_hist;

    // Compute the histograms:
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    // Normalize the result
    normalize(b_hist, b_hist, 0, 100, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, 100, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, 100, NORM_MINMAX, -1, Mat());

    int smooth_size = 5;
    GaussianBlur(b_hist, b_hist, Size(smooth_size, smooth_size), 0, 0, BORDER_CONSTANT);
    GaussianBlur(g_hist, g_hist, Size(smooth_size, smooth_size), 0, 0, BORDER_CONSTANT);
    GaussianBlur(r_hist, r_hist, Size(smooth_size, smooth_size), 0, 0, BORDER_CONSTANT);

    return { b_hist, g_hist, r_hist };
}

void draw_hist(Mat& img) {
    vector<Mat> bgr_planes = calc_hist(img);
    Mat b_hist = bgr_planes[0];
    Mat g_hist = bgr_planes[1];
    Mat r_hist = bgr_planes[2];

    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 400;
    int bin_w = 2;
    int histSize = 256;

    normalize(b_hist, b_hist, 0, 400, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, 400, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, 400, NORM_MINMAX, -1, Mat());

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    // Draw for each channel
    for (int i = 1; i < histSize; i++) {
        cout << b_hist.at<int>(i - 1) << endl;

        line(histImage, Point(bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1))),
            Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, Point(bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1))),
            Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
            Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, Point(bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1))),
            Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
            Scalar(0, 0, 255), 2, 8, 0);
    }

    // Display
    namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
    imshow("calcHist Demo", histImage);
}


bool wasClicked = false;
Point lastPoint;
Mat src;
Mat src_gray;
const string window_name = "img";

void on_mouse(int event, int x, int y, int, void*) {
    if (event != EVENT_LBUTTONDOWN)
        return;

    wasClicked ^= true;

    if (!wasClicked) {
        lastPoint = Point(x, y);
        return;
    }

    Mat copy;
    src.copyTo(copy);
    Rect rect = Rect(lastPoint, Point(x, y));
    Mat roi(src, rect);

    rectangle(copy, rect, Scalar(0, 0, 255));
    imshow(window_name, copy);

    draw_hist(roi);
}


//void find_homogenity(Mat& img, int stripe_width, int stripe_height) {
// for (int i = 0; i < img.rows - stripe_height; i++) {
//     for (int j = 0; j < img.cols - stripe_width; j++) {
//     }
// }
//}

void calc_gradient() {
    GaussianBlur(src_gray, src_gray, Size(3, 3), 0, 0);

    Mat grad_x, grad_y, grad;

    Sobel(src_gray, grad_x, CV_16S, 1, 0);
    convertScaleAbs(grad_x, grad_x);

    Sobel(src_gray, grad_y, CV_16S, 0, 1);
    convertScaleAbs(grad_y, grad_y);

    addWeighted(grad_x, 0.5, grad_y, 0.5, 0, grad);
    imshow("gradient", grad);
}


Mat detected_edges;
int low_threshold;
int ratio = 6;
const string canny_window_name = "canny edge detector";

void canny_callback(int, void*) {
    Mat blurred;
    GaussianBlur(src, blurred, Size(5, 5), 0, 0);
//    src_gray.copyTo(blurred);
    Canny(blurred, detected_edges, low_threshold, low_threshold * ratio, 5);
    Mat result_canny = Mat::zeros(src.rows, src.cols, CV_8U);
    src.copyTo(result_canny, detected_edges);
    imshow(canny_window_name, result_canny);
}

float hist_variance(Mat hist) {
    assert(hist.type() == CV_32F);

    float mean = 0;
    for (int i = 0; i < hist.rows; i++) {
        mean += hist.at<float>(i);
    }

    mean /= hist.rows;

    float variance = 0;
    for (int i = 0; i < hist.rows; i++) {
        variance += (hist.at<float>(i) - mean) * (hist.at<float>(i) - mean);
    }

    return variance / (hist.rows - 1);
}

void stripes() {
    int stripe_width = 5;
    int stripe_height = 21;
    float max_variance = 20.f;

    Mat dst;
    src.copyTo(dst);

    for (int i = 0; i < src.rows - stripe_height - 1; i++) {
        for (int j = 0; j < src.cols - stripe_width - 1; j++) {
            Rect rect = Rect(Point(j, i), Point(j + stripe_width, i + stripe_height));
            Mat roi(src, rect);

            std::vector<Mat> bgr_planes = calc_hist(roi);
            float b_var = hist_variance(bgr_planes[0]);
            float g_var = hist_variance(bgr_planes[1]);
            float r_var = hist_variance(bgr_planes[2]);

            if (b_var < max_variance &&
                    g_var < max_variance &&
                    r_var < max_variance)
            {
                rectangle(dst, rect, Scalar(0, 0, 255));
            }
        }
    }

    imshow("detected stripes", dst);
}

Vec3d operator - (const Vec3d& left, const Vec3d& right) {
    return Vec3d(left[0] - right[0],
            left[1] - right[1],
            left[2] - right[2]);
}

Vec3d operator * (const Vec3d& left, const Vec3d& right) {
    return Vec3d(left[0] * right[0],
            left[1] * right[1],
            left[2] * right[2]);
}

Vec3d correlation(const Mat& vec1, const Mat& vec2) {
    assert(vec1.cols == vec2.cols);
    Mat vec1d, vec2d;
    vec1.convertTo(vec1d, CV_64F);
    vec2.convertTo(vec2d, CV_64F);

    Vec3d mean1(0.);
    Vec3d mean2(0.);
    int n = vec1.cols;

    for (int i = 0; i < n; i++) {
        mean1 += vec1d.at<Vec3d>(i);
        mean2 += vec2d.at<Vec3d>(i);
    }

    mean1 /= n;
    mean2 /= n;

//    cout << mean1 << mean2 << endl;

    Vec3d cov(0.);
    Vec3d disp1(0.);
    Vec3d disp2(0.);

    for (int i = 0; i < n; i++) {
        cov += (vec1d.at<Vec3d>(i) - mean1) * (vec2d.at<Vec3d>(i) - mean2);
        disp1 += (vec1d.at<Vec3d>(i) - mean1) * (vec1d.at<Vec3d>(i) - mean1);
        disp2 += (vec2d.at<Vec3d>(i) - mean2) * (vec2d.at<Vec3d>(i) - mean2);
    }

//    cout << "cov = " << cov << " disp1 = " << disp1 << " disp2 = " << disp2 << endl;

    return Vec3d(
                    cov[0] / (sqrt(disp1[0]) * sqrt(disp2[0])),
                    cov[1] / (sqrt(disp1[1]) * sqrt(disp2[1])),
                    cov[2] / (sqrt(disp1[2]) * sqrt(disp2[2]))
                 );
}

void corr_stripes() {
    int stripe_width = 51;
    int stripe_height = 1;

    Mat dst;
    src.copyTo(dst);

    for (int i = 2; i < src.rows - 2; i++) {
        for (int j = 0; j < src.cols - stripe_width - 1; j++) {
            Rect rect_up1 = Rect(Point(j, i - 1), Point(j + stripe_width, i + stripe_height - 1));
            Rect rect_up2 = Rect(Point(j, i - 2), Point(j + stripe_width, i + stripe_height - 2));

            Rect rect_orig = Rect(Point(j, i), Point(j + stripe_width, i + stripe_height));

            Rect rect_down1 = Rect(Point(j, i + 1), Point(j + stripe_width, i + stripe_height + 1));
            Rect rect_down2 = Rect(Point(j, i + 2), Point(j + stripe_width, i + stripe_height + 2));

            Mat roi_up1(src, rect_up1);
            Mat roi_up2(src, rect_up2);

            Mat roi(src, rect_orig);

            Mat roi_down1(src, rect_down1);
            Mat roi_down2(src, rect_down2);

            Vec3d corr_up1 = correlation(roi_up1, roi);
            Vec3d corr_up2 = correlation(roi_up2, roi);
            Vec3d corr_down1 = correlation(roi_down1, roi);
            Vec3d corr_down2 = correlation(roi_down2, roi);

            Vec3d coeff = corr_up1 * corr_up2 * corr_down1 * corr_down2;

            if (coeff[0] > 0.95 && coeff[1] > 0.95 && coeff[2] > 0.95) {
                Vec3b color(coeff[0] * 255., coeff[1] * 255., coeff[2] * 255.);
                line(dst, Point(j, i), Point(j + stripe_width, i + stripe_height + 1), color);
            }
        }
    }

    Mat dst2 = src * 0.8 + dst * 0.2;

    imshow("correlation", dst2);
}

int main() {
    src = imread("map.png", 1);
    src.copyTo(detected_edges);
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    namedWindow(window_name);
    setMouseCallback(window_name, on_mouse);

    namedWindow(canny_window_name);
    createTrackbar("Min Threshold:", canny_window_name, &low_threshold, 500, canny_callback);
    canny_callback(0, 0);

    imshow(window_name, src);
//    imshow("gray", src_gray);
//    draw_hist(src);

//    stripes();
    corr_stripes();
    waitKey();

    return 0;
}
