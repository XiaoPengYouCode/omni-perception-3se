#include "perception/angle_utils.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {
constexpr double kPi = 3.14159265358979323846;
} // namespace

namespace op3 {

CameraModel make_default_camera_model(CameraPosition camera, int image_width, int image_height) {
  switch (camera) {
  case CameraPosition::kLeftFront:
    return CameraModel{
        .camera = camera,
        .image_width = image_width,
        .image_height = image_height,
        .horizontal_fov_degrees = 90.0,
        .assumed_person_height_m = 1.7,
        .max_range_m = 12.0,
        .mount_x_m = 0.35,
        .mount_y_m = 0.22,
    };
  case CameraPosition::kRightFront:
    return CameraModel{
        .camera = camera,
        .image_width = image_width,
        .image_height = image_height,
        .horizontal_fov_degrees = 90.0,
        .assumed_person_height_m = 1.7,
        .max_range_m = 12.0,
        .mount_x_m = 0.35,
        .mount_y_m = -0.22,
    };
  case CameraPosition::kLeftRear:
    return CameraModel{
        .camera = camera,
        .image_width = image_width,
        .image_height = image_height,
        .horizontal_fov_degrees = 90.0,
        .assumed_person_height_m = 1.7,
        .max_range_m = 12.0,
        .mount_x_m = -0.35,
        .mount_y_m = 0.22,
    };
  case CameraPosition::kRightRear:
    return CameraModel{
        .camera = camera,
        .image_width = image_width,
        .image_height = image_height,
        .horizontal_fov_degrees = 90.0,
        .assumed_person_height_m = 1.7,
        .max_range_m = 12.0,
        .mount_x_m = -0.35,
        .mount_y_m = -0.22,
    };
  }

  throw std::invalid_argument("unknown camera position");
}

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

double estimate_person_range(const cv::Rect& bbox, const CameraModel& model) {
  if (bbox.height <= 0 || model.image_height <= 0 || model.image_width <= 0 ||
      model.assumed_person_height_m <= 0.0 || model.horizontal_fov_degrees <= 0.0) {
    throw std::invalid_argument("invalid camera model or bbox");
  }

  const double focal_length = focal_length_pixels(model.image_width, model.horizontal_fov_degrees);
  const double unclamped_range =
      (model.assumed_person_height_m * focal_length) / static_cast<double>(bbox.height);
  return std::clamp(unclamped_range, 0.1, model.max_range_m);
}

double focal_length_pixels(int image_width, double horizontal_fov_degrees) {
  if (image_width <= 0 || horizontal_fov_degrees <= 0.0) {
    throw std::invalid_argument("invalid image width or horizontal fov");
  }

  const double horizontal_fov_radians = horizontal_fov_degrees * (kPi / 180.0);
  return (static_cast<double>(image_width) * 0.5) / std::tan(horizontal_fov_radians * 0.5);
}

double project_person_height_pixels(double range_m, const CameraModel& model) {
  if (range_m <= 0.0 || model.assumed_person_height_m <= 0.0 || model.image_width <= 0 ||
      model.horizontal_fov_degrees <= 0.0) {
    throw std::invalid_argument("invalid range or camera model");
  }

  const double focal_length = focal_length_pixels(model.image_width, model.horizontal_fov_degrees);
  return (model.assumed_person_height_m * focal_length) / range_m;
}

cv::Point2d body_frame_point_from_polar(double range_m, double angle_degrees) {
  const double radians = angle_degrees * (kPi / 180.0);
  return {range_m * std::cos(radians), range_m * std::sin(radians)};
}

cv::Point2d rotate_point(double x_m, double y_m, double yaw_degrees) {
  const double radians = yaw_degrees * (kPi / 180.0);
  const double cos_yaw = std::cos(radians);
  const double sin_yaw = std::sin(radians);
  return {
      (x_m * cos_yaw) - (y_m * sin_yaw),
      (x_m * sin_yaw) + (y_m * cos_yaw),
  };
}

cv::Point2d body_to_world_point(double x_m, double y_m, const RobotPose& robot_pose) {
  const cv::Point2d rotated = rotate_point(x_m, y_m, robot_pose.yaw_degrees);
  return {robot_pose.x_m + rotated.x, robot_pose.y_m + rotated.y};
}

cv::Point2d world_to_body_point(double x_m, double y_m, const RobotPose& robot_pose) {
  return rotate_point(x_m - robot_pose.x_m, y_m - robot_pose.y_m, -robot_pose.yaw_degrees);
}

double body_frame_angle_from_point(double x_m, double y_m) {
  return normalize_angle(std::atan2(y_m, x_m) * (180.0 / kPi));
}

double body_frame_range_from_point(double x_m, double y_m) {
  return std::hypot(x_m, y_m);
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

} // namespace op3
