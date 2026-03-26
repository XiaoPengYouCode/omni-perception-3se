#pragma once

#include <string>
#include <vector>

#include "perception/detector.hpp"
#include "perception/mock_person_detector.hpp"
#include "perception/types.hpp"

namespace op3 {

/**
 * Detector stub that pretends to be a yolo26-nano model while delegating to the mock detector.
 */
class Yolo26NanoDetector final : public Detector {
public:
  /**
   * Builds a yolo26-nano detector for a specific camera mount.
   */
  explicit Yolo26NanoDetector(CameraModel camera_model, std::string detector_name);

  Yolo26NanoDetector(const Yolo26NanoDetector&) = delete;
  Yolo26NanoDetector& operator=(const Yolo26NanoDetector&) = delete;

  /**
   * Delegates inference to the mock detector and prefixes ids with the logical detector name.
   */
  std::vector<Detection> detect(const cv::Mat& image) override;

  /**
   * Returns the camera model used to configure this detector.
   */
  const CameraModel& camera_model() const;

private:
  CameraModel camera_model_;
  std::string detector_name_;
  MockPersonDetector delegate_;
};

} // namespace op3
