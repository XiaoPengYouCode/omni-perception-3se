#include "simulation/simulation_engine.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "perception/angle_utils.hpp"
#include "perception/scene_labels.hpp"

namespace op3 {

namespace {

constexpr double kPi = 3.14159265358979323846;

double clamp_angle(double degrees) {
  double normalized = std::fmod(degrees + 180.0, 360.0);
  if (normalized < 0.0) {
    normalized += 360.0;
  }
  return normalized - 180.0;
}

double rotate_radians(double degrees) {
  return degrees * (kPi / 180.0);
}

double projected_person_height_pixels(const SimulationConfig& config, double range_m) {
  if (range_m <= 0.0) {
    return 0.0;
  }

  const double focal_length =
      focal_length_pixels(config.camera_image_size.width, config.camera_horizontal_fov_degrees);
  return (config.assumed_person_height_m * focal_length / range_m) * config.person_pixel_scale;
}

} // namespace

SimulationEngine::SimulationEngine(SimulationConfig config)
    : config_(std::move(config)),
      robot_({.position = {config_.map_size.width * 0.5, config_.map_size.height * 0.5},
              .yaw_degrees = 0.0}),
      pedestrians_() {
  pedestrians_.reserve(config_.pedestrian_waypoints.size());
  for (const SimulationConfig::PedestrianWaypoint& waypoint : config_.pedestrian_waypoints) {
    if (waypoint.waypoints.empty()) {
      continue;
    }
    pedestrians_.push_back(PedestrianState{
        .id = waypoint.id,
        .position = waypoint.waypoints.front(),
        .waypoints = waypoint.waypoints,
        .target_index = 1 % waypoint.waypoints.size(),
        .speed_mps =
            waypoint.speed_mps > 0.0 ? waypoint.speed_mps : config_.default_pedestrian_speed_mps,
    });
  }
}

void SimulationEngine::update(double delta_seconds, const SimulationControls& controls) {
  const double forward = (controls.move_forward ? 1.0 : 0.0) - (controls.move_backward ? 1.0 : 0.0);
  const double strafe = (controls.move_right ? 1.0 : 0.0) - (controls.move_left ? 1.0 : 0.0);
  const double rotation = (controls.rotate_right ? 1.0 : 0.0) - (controls.rotate_left ? 1.0 : 0.0);

  const double yaw_rad = rotate_radians(robot_.yaw_degrees);
  const double forward_delta = forward * config_.robot_speed_mps * delta_seconds;
  const double strafe_delta = strafe * config_.robot_strafe_speed_mps * delta_seconds;
  robot_.position.x += forward_delta * std::cos(yaw_rad) - strafe_delta * std::sin(yaw_rad);
  robot_.position.y += forward_delta * std::sin(yaw_rad) + strafe_delta * std::cos(yaw_rad);
  robot_.yaw_degrees = clamp_angle(robot_.yaw_degrees +
                                   rotation * config_.robot_turn_deg_per_second * delta_seconds);

  robot_.position.x = std::clamp(robot_.position.x, 0.0, config_.map_size.width);
  robot_.position.y = std::clamp(robot_.position.y, 0.0, config_.map_size.height);

  advance_pedestrians(delta_seconds);
}

void SimulationEngine::advance_pedestrians(double delta_seconds) {
  for (PedestrianState& ped : pedestrians_) {
    if (ped.waypoints.empty()) {
      continue;
    }
    const cv::Point2d target = ped.waypoints[ped.target_index];
    cv::Point2d delta = target - ped.position;
    const double distance = std::hypot(delta.x, delta.y);
    if (distance <= 1e-3) {
      ped.target_index = (ped.target_index + 1) % ped.waypoints.size();
      continue;
    }
    const double travel = ped.speed_mps * delta_seconds;
    if (travel >= distance) {
      ped.position = target;
      ped.target_index = (ped.target_index + 1) % ped.waypoints.size();
      continue;
    }
    const double ratio = travel / distance;
    ped.position.x += delta.x * ratio;
    ped.position.y += delta.y * ratio;
  }
}

std::vector<FrameInput> SimulationEngine::render_frames() const {
  std::vector<FrameInput> frames;
  frames.reserve(config_.camera_mounts.size());
  for (const SimulationConfig::CameraMount& mount : config_.camera_mounts) {
    cv::Mat image(config_.camera_image_size, CV_8UC3, cv::Scalar(235, 235, 235));

    const cv::Point2d camera_pos =
        robot_.position + rotate_offset(mount.mount_offset, robot_.yaw_degrees);
    const double mount_yaw = clamp_angle(robot_.yaw_degrees + mount.yaw_offset_degrees);

    for (const PedestrianState& ped : pedestrians_) {
      cv::Point2d relative = ped.position - camera_pos;
      double ped_angle = body_frame_angle_from_point(relative.x, relative.y);
      double angle_offset = clamp_angle(ped_angle - mount_yaw);
      if (std::abs(angle_offset) > (config_.camera_horizontal_fov_degrees * 0.5)) {
        continue;
      }

      const double range = body_frame_range_from_point(relative.x, relative.y);
      if (range > config_.max_person_range_m) {
        continue;
      }
      const double normalized_x =
          std::clamp((angle_offset + (config_.camera_horizontal_fov_degrees * 0.5)) /
                         config_.camera_horizontal_fov_degrees,
                     0.0, 1.0);
      const double screen_x = normalized_x * static_cast<double>(image.cols);
      const double size_px =
          std::clamp(projected_person_height_pixels(config_, std::max(range, 0.1)), 12.0,
                     static_cast<double>(image.rows) * 0.65);
      const double width = size_px * 0.45;
      const double height = size_px;
      const double bottom_y = static_cast<double>(image.rows) - 10.0;
      const cv::Point2d top_left(screen_x - (width * 0.5), bottom_y - height);
      const cv::Rect bbox(static_cast<int>(std::clamp(top_left.x, 0.0, image.cols - width)),
                          static_cast<int>(std::clamp(top_left.y, 0.0, image.rows - height)),
                          static_cast<int>(std::clamp(width, 2.0, image.cols - top_left.x)),
                          static_cast<int>(std::clamp(height, 2.0, image.rows - top_left.y)));
      const int fill_gray = scene_label_grayscale(ped.id);
      cv::rectangle(image, bbox, cv::Scalar(fill_gray, fill_gray, fill_gray), cv::FILLED);
      cv::rectangle(image, bbox, cv::Scalar(245, 245, 245), 2);
    }

    frames.push_back(FrameInput{.camera = mount.camera, .image = image});
  }

  return frames;
}

RobotState SimulationEngine::robot_state() const {
  return robot_;
}

std::vector<PedestrianState> SimulationEngine::pedestrian_states() const {
  return pedestrians_;
}

cv::Point2d SimulationEngine::rotate_offset(const cv::Point2d& offset, double yaw_degrees) const {
  const double rad = rotate_radians(yaw_degrees);
  const double cos_yaw = std::cos(rad);
  const double sin_yaw = std::sin(rad);
  return cv::Point2d{offset.x * cos_yaw - offset.y * sin_yaw,
                     offset.x * sin_yaw + offset.y * cos_yaw};
}

SimulationConfig SimulationEngine::default_config() {
  SimulationConfig config;
  config.camera_mounts = {
      {CameraPosition::kLeftFront, 45.0, {0.35, 0.22}},
      {CameraPosition::kRightFront, -45.0, {0.35, -0.22}},
      {CameraPosition::kLeftRear, 135.0, {-0.35, 0.22}},
      {CameraPosition::kRightRear, -135.0, {-0.35, -0.22}},
  };
  config.pedestrian_waypoints = {
      {"Hero", {{5.0, 5.0}, {10.0, 5.0}, {10.0, 10.0}, {5.0, 10.0}}, 3.6},
      {"Engineer", {{2.0, 12.0}, {6.0, 14.0}, {8.0, 8.0}, {4.0, 6.0}}, 2.8},
      {"infantry", {{15.0, 8.0}, {14.0, 12.0}, {12.0, 16.0}, {10.0, 10.0}}, 4.0},
      {"sentry", {{16.0, 4.0}, {17.0, 7.0}, {14.0, 9.0}, {12.0, 4.0}}, 3.2},
  };
  return config;
}

} // namespace op3
