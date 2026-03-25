#pragma once

#include "perception/detector.hpp"

namespace op3 {

/**
 * Demo detector that treats a dark blob as a single detected person.
 */
class MockPersonDetector final : public Detector {
public:
  /**
   * Produces a deterministic fake person detection from a synthetic demo frame.
   */
  std::vector<Detection> detect(const cv::Mat& image) override;
};

} // namespace op3
