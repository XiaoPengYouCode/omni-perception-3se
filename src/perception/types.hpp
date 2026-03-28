#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace op3 {

/**
 * Enumerates the fixed physical mounting positions of the four cameras.
 */
enum class CameraPosition {
  kLeftFront,
  kRightFront,
  kLeftRear,
  kRightRear,
};

/**
 * Raw frame payload used by local demo code before it enters the threaded pipeline.
 */
struct FrameInput {
  CameraPosition camera;
  cv::Mat image;
};

/**
 * Low-level detector output before conversion into the shared robot-centric view.
 */
struct Detection {
  std::string id;
  std::string label;
  cv::Rect bbox;
  float confidence = 0.0F;
};

/**
 * Basic per-camera geometry and rendering assumptions shared by simulation and perception.
 */
struct CameraModel {
  CameraPosition camera;
  int image_width = 0;
  int image_height = 0;
  double horizontal_fov_degrees = 90.0;
  double assumed_person_height_m = 1.7;
  double assumed_person_width_m = 0.765;
  double max_range_m = 12.0;
  double mount_x_m = 0.0;
  double mount_y_m = 0.0;
};

/**
 * Robot pose expressed in world coordinates.
 */
struct RobotPose {
  double x_m = 0.0;
  double y_m = 0.0;
  double yaw_degrees = 0.0;
};

/**
 * A single person observation already expressed as a body-frame angle.
 */
struct PersonReport {
  std::string id;
  std::string label;
  double angle = 0.0;
  double range_m = 0.0;
  double x_m = 0.0;
  double y_m = 0.0;
  double world_x_m = 0.0;
  double world_y_m = 0.0;
  CameraPosition camera;
};

/**
 * Message pushed into a camera worker input queue.
 */
struct FrameMessage {
  CameraPosition camera;
  std::uint64_t frame_id;
  std::chrono::steady_clock::time_point timestamp;
  RobotPose robot_pose;
  cv::Mat image;
};

/**
 * Message emitted by a camera worker after per-camera inference completes.
 */
struct DetectionMessage {
  CameraPosition camera;
  std::uint64_t frame_id;
  std::chrono::steady_clock::time_point timestamp;
  RobotPose robot_pose;
  std::vector<PersonReport> person;
};

/**
 * A fused track state describing one persistent person around the robot.
 */
struct TrackedPerson {
  std::string track_id;
  std::string label;
  double x_m = 0.0;
  double y_m = 0.0;
  double world_x_m = 0.0;
  double world_y_m = 0.0;
  double vx_mps = 0.0;
  double vy_mps = 0.0;
  double range_m = 0.0;
  double radius_m = 0.0;
  double angle = 0.0;
  double angle_velocity = 0.0;
  double confidence = 0.0;
  std::vector<CameraPosition> sources;
  std::chrono::steady_clock::time_point last_update;
};

/**
 * Snapshot of the current fused world state that gets serialized to JSON.
 */
struct PipelineOutput {
  std::uint64_t sequence_id;
  std::int64_t timestamp_ms;
  std::vector<TrackedPerson> person;
};

} // namespace op3
