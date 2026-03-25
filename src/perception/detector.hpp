#pragma once

#include <vector>

#include "perception/types.hpp"

namespace op3 {

/**
 * Interface for any per-camera person detector backend.
 */
class Detector {
 public:
  /**
   * Virtual destructor for polymorphic use.
   */
  virtual ~Detector() = default;

  /**
   * Runs detector inference on a single image.
   *
   * @param image Source frame.
   * @return Zero or more raw detections in image coordinates.
   */
  virtual std::vector<Detection> detect(const cv::Mat& image) = 0;
};

}  // namespace op3
