#pragma once

#include <memory>

#include "perception/detector.hpp"
#include "perception/types.hpp"

namespace op3 {

/**
 * Creates detector instances for every supported backend.
 */
class DetectorFactory {
public:
  /**
   * Builds a yolo26-nano detector configured for a specific camera model.
   */
  static std::unique_ptr<Detector> create_yolo26_nano(CameraModel camera_model);
};

} // namespace op3
