#include "perception/yolo26_nano_detector.hpp"

#include <utility>

namespace op3 {

Yolo26NanoDetector::Yolo26NanoDetector(CameraModel camera_model, std::string detector_name)
    : camera_model_(std::move(camera_model)), detector_name_(std::move(detector_name)) {}

const CameraModel& Yolo26NanoDetector::camera_model() const {
  return camera_model_;
}

std::vector<Detection> Yolo26NanoDetector::detect(const cv::Mat& image) {
  std::vector<Detection> detections = delegate_.detect(image);
  if (detector_name_.empty()) {
    return detections;
  }

  for (Detection& detection : detections) {
    detection.id = detector_name_ + "-" + detection.id;
  }

  return detections;
}

} // namespace op3
