#include "perception/angle_utils.hpp"

#include <cmath>
#include <stdexcept>

namespace op3 {

double camera_base_angle(CameraPosition camera) {
  switch (camera) {
    case CameraPosition::kLeftFront:
      return 45.0;
    case CameraPosition::kRightFront:
      return -45.0;
    case CameraPosition::kLeftRear:
      return 135.0;
    case CameraPosition::kRightRear:
      return -135.0;
  }

  throw std::invalid_argument("unknown camera position");
}

double normalize_angle(double angle_degrees) {
  // Normalize through modulo arithmetic so every downstream component sees one canonical range.
  double normalized = std::fmod(angle_degrees + 180.0, 360.0);
  if (normalized < 0.0) {
    normalized += 360.0;
  }
  return normalized - 180.0;
}

double compute_person_angle(CameraPosition camera, const cv::Rect& bbox, int image_width,
                            double horizontal_fov_degrees) {
  if (image_width <= 0) {
    throw std::invalid_argument("image width must be positive");
  }

  // Interpret the box center as a horizontal bearing offset from the camera optical axis.
  const double center_x = static_cast<double>(bbox.x) + (static_cast<double>(bbox.width) / 2.0);
  const double normalized_x = center_x / static_cast<double>(image_width);
  const double offset = (normalized_x - 0.5) * horizontal_fov_degrees;
  return normalize_angle(camera_base_angle(camera) + offset);
}

std::string camera_position_to_string(CameraPosition camera) {
  switch (camera) {
    case CameraPosition::kLeftFront:
      return "left_front";
    case CameraPosition::kRightFront:
      return "right_front";
    case CameraPosition::kLeftRear:
      return "left_rear";
    case CameraPosition::kRightRear:
      return "right_rear";
  }

  throw std::invalid_argument("unknown camera position");
}

}  // namespace op3
