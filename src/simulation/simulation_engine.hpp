#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "perception/types.hpp"

namespace op3 {

/**
 * Configures the built-in simulator for a simple 2D scene.
 */
struct SimulationConfig {
  double tick_seconds = 0.016;
  cv::Size2d map_size{20.0, 20.0};
  double robot_speed_mps = 4.8;
  double robot_strafe_speed_mps = 3.2;
  double robot_turn_deg_per_second = 360.0;
  double default_pedestrian_speed_mps = 3.6;
  cv::Size camera_image_size{960, 540};
  double camera_horizontal_fov_degrees = 90.0;
  double assumed_person_height_m = 1.7;
  double person_pixel_scale = 1.0;
  double max_person_range_m = 12.0;

  struct CameraMount {
    CameraPosition camera;
    double yaw_offset_degrees;
    cv::Point2d mount_offset;
  };

  struct PedestrianWaypoint {
    std::string id;
    std::vector<cv::Point2d> waypoints;
    double speed_mps;
  };

  std::vector<CameraMount> camera_mounts;
  std::vector<PedestrianWaypoint> pedestrian_waypoints;
};

/**
 * Represents the current robot pose inside the simulator.
 */
struct RobotState {
  cv::Point2d position{0.0, 0.0};
  double yaw_degrees = 0.0;
};

/**
 * Input commands that gate motion across frames.
 */
struct SimulationControls {
  bool move_forward = false;
  bool move_backward = false;
  bool move_left = false;
  bool move_right = false;
  bool rotate_left = false;
  bool rotate_right = false;
};

/**
 * Internal state for each simulated person.
 */
struct PedestrianState {
  std::string id;
  cv::Point2d position;
  std::vector<cv::Point2d> waypoints;
  std::size_t target_index = 0;
  double speed_mps = 0.0;
};

/**
 * Simple in-process simulator that updates robot + pedestrian states and renders camera frames.
 */
class SimulationEngine {
public:
  explicit SimulationEngine(SimulationConfig config);

  /**
   * Steps the simulation forward by the given delta and applies the control signals.
   */
  void update(double delta_seconds, const SimulationControls& controls);

  /**
   * Renders four camera frames according to the current robot/pedestrian poses.
   */
  std::vector<FrameInput> render_frames() const;

  /**
   * Returns the current robot pose.
   */
  RobotState robot_state() const;

  /**
   * Returns a copy of the pedestrian states for inspection.
   */
  std::vector<PedestrianState> pedestrian_states() const;

  /**
   * Default configuration tuned for the demo.
   */
  static SimulationConfig default_config();

private:
  void advance_pedestrians(double delta_seconds);
  cv::Point2d rotate_offset(const cv::Point2d& offset, double yaw_degrees) const;
  const SimulationConfig config_;
  RobotState robot_;
  std::vector<PedestrianState> pedestrians_;
};

} // namespace op3
