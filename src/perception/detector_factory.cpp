#include "perception/detector_factory.hpp"

#include <string>
#include <utility>

#include "perception/angle_utils.hpp"
#include "perception/yolo26_nano_detector.hpp"

namespace op3 {

std::unique_ptr<Detector> DetectorFactory::create_yolo26_nano(CameraModel camera_model) {
  const std::string detector_name = "yolo26-nano-" + camera_position_to_string(camera_model.camera);
  return std::make_unique<Yolo26NanoDetector>(std::move(camera_model), std::move(detector_name));
}

} // namespace op3
