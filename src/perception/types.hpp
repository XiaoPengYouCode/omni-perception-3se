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
  cv::Rect bbox;
  float confidence;
};

/**
 * A single person observation already expressed as a body-frame angle.
 */
struct PersonReport {
  std::string id;
  double angle;
  CameraPosition camera;
};

/**
 * Message pushed into a camera worker input queue.
 */
struct FrameMessage {
  CameraPosition camera;
  std::uint64_t frame_id;
  std::chrono::steady_clock::time_point timestamp;
  cv::Mat image;
};

/**
 * Message emitted by a camera worker after per-camera inference completes.
 */
struct DetectionMessage {
  CameraPosition camera;
  std::uint64_t frame_id;
  std::chrono::steady_clock::time_point timestamp;
  std::vector<PersonReport> person;
};

/**
 * A fused track state describing one persistent person around the robot.
 */
struct TrackedPerson {
  std::string track_id;
  double angle;
  double angle_velocity;
  double confidence;
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
