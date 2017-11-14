#include <math.h>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <set>

#include "edges.h"

//const double MIN_VAL = 1e-5;

Ray findBestDirectionAt(int x, int y, const cv::Mat& image);

cv::Mat* orig;

Corner findBestCornerAt(int x, int y, const cv::Mat& image, double* bestScoreOut = nullptr) {
  double bestScore;
  Ray bestDirection = findBestDirectionAt(x, y, image);
  Corner result(x, y, bestDirection.alpha());
  bestScore = -1./0.;
  for (double beta = 0; beta < 2 * M_PI; beta += M_PI / 180) {
    Corner current(x, y, bestDirection.alpha(), beta);
    double score = current.score(image);
    if (score > bestScore) {
      result = current;
      bestScore = score;
    }
  }
  if (bestScoreOut)
    *bestScoreOut = bestScore;
  return result;
}

Ray findBestDirectionAt(int x, int y, const cv::Mat& image) {
  Ray bestDirection(x, y, 0);
  double bestScore = -1./0.;
  for (int side = 0; side < 2; side++) {
    for (double alpha = 0; alpha < 2 * M_PI; alpha += M_PI / 180) {
      Ray current(x, y, alpha, bool(side));
      double score = current.score(image);
      if (score > bestScore) {
        bestDirection = current;
        bestScore = score;
      }
    }
  }
  return bestDirection;
}

void drawBestCornerAt(int x, int y, const cv::Mat& image, cv::Mat& orig, cv::Mat& to) {
  double bestScore;
  Corner best = findBestCornerAt(x, y, image, &bestScore);

  if (bestScore > 0.4) {
    best.draw(to, best.score(image));

    std::ostringstream fileName;
    fileName << "corner@" << x << "," << y << ".png";
    cv::Mat copy_img = orig.clone();
    best.draw(copy_img, bestScore);
    cv::imwrite(fileName.str(), copy_img);
  }
}


LineSegment fitLineSegmentAtSlow(int x, int y, const cv::Mat& image) {
  LineSegment result(x, y, x + 1, y);
  const int side = std::min(100, (int)(std::max(image.cols, image.rows) * sqrt(2)));
  double bestScore = result.score(image);
  for (int i = 0; i < 360; i+=1) {
    double alpha = 2. * M_PI * i / 360;
    Ray direction(x, y, alpha);
    for (int l = 2; l < side / 2; l++) {
      LineSegment segment(direction.alpha(), direction.beta(), direction.t_0(), direction.t_0() + l);
      double value = segment.score(image);
      if (bestScore < value) {
        result = segment;
        bestScore = value;
      }
    }
  }
  return result;
}

LineSegment fitLineSegmentAtSlowBiDi(int x, int y, const cv::Mat& image) {
  LineSegment result(x, y, x + 1, y);
  const int side = std::min(100, (int)(std::max(image.cols, image.rows) * sqrt(2)));
  double bestScore = result.score(image);
  for (int i = 0; i < 360; i+=1) {
    double alpha = 2. * M_PI * i / 360;
    Ray direction(x, y, alpha);
    LineSegment bestLineSegmentForAlpha(direction, 0);
    double bestScoreForAlpha = bestLineSegmentForAlpha.score(image);

    int bestT_0 = 0;
    for (int l = 1; l < side / 2; l++) {
      LineSegment segment(direction, l);
      double value = segment.score(image);
      if (bestScoreForAlpha < value) {
        bestLineSegmentForAlpha = segment;
        bestScoreForAlpha = value;
        bestT_0 = l;
      }
    }
    for (int k = 1; k < side / 2; k++) {
      LineSegment segment(direction.alpha(), direction.beta(), direction.t_0() - k, direction.t_0() + bestT_0);
      double value = segment.score(image);
      if (bestScoreForAlpha < value) {
        bestLineSegmentForAlpha = segment;
        bestScoreForAlpha = value;
      }
    }
    if (bestScoreForAlpha > bestScore) {
      result = bestLineSegmentForAlpha;
      bestScore = bestScoreForAlpha;
    }
  }
  return result;
}

double fitPolygonAt(int x, int y, const cv::Mat& image, std::vector<LineSegment>& result) {
  LineSegment first = fitLineSegmentAtSlow(x, y, image);
  double score = first.score(image);
  std::cout << first << std::endl;
  result.push_back(first);
  while (true) {
    LineSegment edge = fitLineSegmentAtSlow(result.rbegin()->x_1(), result.rbegin()->y_1(), image);
    LineSegment closing(result.rbegin()->x_1(), result.rbegin()->y_1(), result.begin()->x_0(), result.begin()->y_0());
    std::cout << edge << " score: " << edge.score(image) << " closing score: " << closing.score(image) << std::endl;
    if (closing.score(image) + 5 > edge.score(image) || std::find(result.begin(), result.end(), edge) != result.end()) {
      result.push_back(closing);
      score += closing.score(image);
      break;
    }
//    else if (result.size() > 100) {
//      return -1./0.;
//    }
    result.push_back(edge);
    score += edge.score(image);
  }
  return score;
}

void drawBestPolygonAt(int x, int y, const cv::Mat& image, cv::Mat& to) {
  std::vector<LineSegment> result;
  double score = fitPolygonAt(x, y, image, result);
//  if (score < 0 || result.size() < 3)
//    return;
//
  cv::Mat copy_img = to.clone();
  std::cout << " final score: " << score << std::endl;
  for (const LineSegment& edge : result) {
    std::cout << "LineSegment: " << edge << " score: " << edge.score(image) << std::endl;
    edge.draw(copy_img, 1);
  }
  std::ostringstream fileName;
  fileName << "polygon@" << x << "," << y << ".png";
  cv::imwrite(fileName.str(), copy_img);
}


int main() {
  using namespace std;
  using namespace cv;

  cv::Mat img_gray;
  cv::Mat img_norm;
  cv::Mat copy_img;

//  cv::Mat img = imread("../results/hog_10K_small.png");
//  int x = 10, y = 4;
  cv::Mat img = imread("../results/hog_10K.png");
  int x = 142, y = 81;
//  cv::Mat img = imread("../results/hog10k_circles_max.png");
//  int x = 454, y = 275;
//  cv::Mat img = imread("../results/hog_circles_10K.png");
//  int x = 142, y = 81;

  orig = &img;
  cvtColor(img, img_gray, COLOR_BGR2GRAY);
  normalize(img_gray, img_norm, 1, 0, NORM_INF, CV_32F);
  normalize(img_norm, img_gray, 0, 255, NORM_MINMAX, CV_8U);
  cvtColor(img_gray, copy_img, COLOR_GRAY2BGR);

//  drawBestPolygonAt(x, y, img_norm, copy_img);
//  drawBestCornerAt(x, y, img_norm, img, copy_img);
//  std::vector<Corner> corners;
//  for (int x = 90; x < 190; x++) {
//    for (int y = 60; y < 160; y++) {
//  Corner best(0, 0, 0);
//
//  for (int x = 0; x < img.cols; x++) {
//    for (int y = 0; y < img.rows; y++) {
//      const Corner& corner = findBestCorner(x, y, img_norm);
//      corners.push_back(corner);
//      if (corner.score(img_norm) > best.score(img_norm)) {
//        best = corner;
//      }
//    }
//    cout << "Line: " << x << " processed" << endl;
//  }

  std::vector<LineSegment> result;
  Ray current = findBestDirectionAt(x, y, img_norm);

  int index = 0;
  while (true) {
    double bestDistance = 1;
    double bestDistanceScore = -1./0;
    Corner best;
//    Mat temp = img.clone();
//    current.draw(temp, 1);
//    cv::imwrite("ray.png", temp);
    for (int d = 1; d < 100; d++) {
      LineSegment variant(current, d);
      if (variant == LineSegment(141, 84, 224, 109))
        std::cout << "Hello" << std::endl;
      double lineScore = variant.score(img_norm);
      for (int outer = 0; outer < 2; outer++) {
        for (double beta = 0; beta < 2 * M_PI; beta += M_PI / 180) {
          Corner next(current.x(d), current.y(d), current.alpha() + M_PI, beta, bool(outer));
          double nbeta = fabs(normalizeAngle(beta));
          double anglePrior = exp(-fabs(nbeta - M_PI_2));
          double angleScore = next.score(img_norm);
          double score = angleScore * lineScore * anglePrior;
          if (bestDistanceScore < score) {
//            if (score > 0.9) {
//              Mat temp = img.clone();
//              next.draw(temp, 1);
//              cv::imwrite("temp.png", temp);
//            }

            bestDistanceScore = score;
            bestDistance = d;
            best = next;
          }
        }
      }
    }
    if (++index > 100)
      break;
    const LineSegment& edge = LineSegment(current, bestDistance);
    result.push_back(edge);
    {
      std::cout << "LineSegment: " << edge << " score: " << edge.score(img_norm) << std::endl;
      edge.draw(copy_img, 1);
      std::ostringstream fileName;
      fileName << "polygon@" << x << "," << y << ".png";
      cv::imwrite(fileName.str(), copy_img);
    }

    current = best.dir2();
  }
  for (const LineSegment& edge : result) {
    std::cout << "LineSegment: " << edge << " score: " << edge.score(img_norm) << std::endl;
    edge.draw(copy_img, 1);
  }
  std::ostringstream fileName;
  fileName << "polygon@" << x << "," << y << ".png";
  cv::imwrite(fileName.str(), copy_img);


//  std::sort(corners.begin(), corners.end(), [&img_norm](const Corner& left, const Corner& right) {
//      return left.score(img_norm) > right.score(img_norm);
//  });
//
//
//  auto it = corners.begin();
//  while (it != corners.end()) {
//    auto rIt = corners.begin();
//    bool hasBetter = false;
//    while (rIt != it) {
//      if (it->overlap(*rIt)) {
//        hasBetter = true;
//        break;
//      }
//      rIt++;
//    }
//    if (hasBetter)
//      it = corners.erase(it);
//    else it++;
//  }
//
//  for (int i = 0; i < 100; i++) {
//    corners[i].draw(copy_img, 10./(i + 10));
//    std::cout << corners[i] << " score: " << corners[i].score(img_norm) << std::endl;
////    corners[i].draw(copy_img, 1.0/i);
//  }

  cv::imwrite("output.png", copy_img);

  return 0;
}
