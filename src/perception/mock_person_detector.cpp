#include "perception/mock_person_detector.hpp"

#include <opencv2/imgproc.hpp>

namespace op3 {

std::vector<Detection> MockPersonDetector::detect(const cv::Mat& image) {
  if (image.empty()) {
    return {};
  }

  // Convert to grayscale so the demo detector only has to reason about intensity.
  cv::Mat grayscale;
  if (image.channels() == 1) {
    grayscale = image;
  } else {
    cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
  }

  // The demo convention is simple: dark pixels represent a person silhouette.
  cv::Mat thresholded;
  cv::threshold(grayscale, thresholded, 80, 255, cv::THRESH_BINARY_INV);

  std::vector<cv::Point> points;
  cv::findNonZero(thresholded, points);
  if (points.empty()) {
    return {};
  }

  // Collapse the silhouette into a single bounding box to mimic a detector output.
  const cv::Rect bbox = cv::boundingRect(points);
  return {Detection{.id = "1", .bbox = bbox, .confidence = 0.95F}};
}

}  // namespace op3
