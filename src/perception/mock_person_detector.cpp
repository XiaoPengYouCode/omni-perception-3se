#include "perception/mock_person_detector.hpp"

#include <algorithm>

#include <opencv2/imgproc.hpp>

#include "perception/scene_labels.hpp"

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

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(thresholded, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) {
    return {};
  }

  std::vector<cv::Rect> boxes;
  boxes.reserve(contours.size());
  for (const std::vector<cv::Point>& contour : contours) {
    const cv::Rect box = cv::boundingRect(contour);
    if (box.width <= 0 || box.height <= 0) {
      continue;
    }
    boxes.push_back(box);
  }

  std::sort(boxes.begin(), boxes.end(),
            [](const cv::Rect& lhs, const cv::Rect& rhs) { return lhs.x < rhs.x; });

  std::vector<Detection> detections;
  detections.reserve(boxes.size());
  for (std::size_t index = 0; index < boxes.size(); ++index) {
    const cv::Rect clipped_box = boxes[index] & cv::Rect(0, 0, grayscale.cols, grayscale.rows);
    const cv::Scalar mean_intensity = cv::mean(grayscale(clipped_box));
    detections.push_back(Detection{
        .id = std::to_string(index + 1),
        .label = decode_scene_label_from_grayscale(mean_intensity[0]),
        .bbox = boxes[index],
        .confidence = 0.95F,
    });
  }
  return detections;
}

} // namespace op3
