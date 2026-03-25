#include "perception/fusion_tracker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include <fmt/format.h>

#include "perception/angle_utils.hpp"

namespace op3 {

namespace {

/**
 * Checks whether a track already records a given source camera.
 */
bool contains_camera(const std::vector<CameraPosition>& cameras, CameraPosition camera) {
  return std::find(cameras.begin(), cameras.end(), camera) != cameras.end();
}

/**
 * Converts steady_clock timestamps into millisecond integers for JSON snapshots.
 */
std::int64_t to_epoch_milliseconds(std::chrono::steady_clock::time_point timestamp) {
  return static_cast<std::int64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(timestamp.time_since_epoch()).count());
}

} // namespace

FusionTracker::FusionTracker(BlockingQueue<DetectionMessage>& detection_queue,
                             double association_gate_degrees, double smoothing_gain)
    : detection_queue_(detection_queue), association_gate_degrees_(association_gate_degrees),
      smoothing_gain_(smoothing_gain) {}

void FusionTracker::start() {
  thread_ = std::thread([this] { run(); });
}

void FusionTracker::stop() {
  detection_queue_.close();
}

void FusionTracker::join() {
  if (thread_.joinable()) {
    thread_.join();
  }
}

PipelineOutput FusionTracker::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);

  PipelineOutput output{
      .sequence_id = latest_sequence_id_,
      .timestamp_ms = to_epoch_milliseconds(latest_timestamp_),
      .person = {},
  };

  output.person.reserve(tracks_.size());
  for (const TrackState& track : tracks_) {
    output.person.push_back(TrackedPerson{
        .track_id = track.track_id,
        .angle = track.angle,
        .angle_velocity = track.angle_velocity,
        .confidence = track.confidence,
        .sources = track.sources,
        .last_update = track.last_update,
    });
  }

  return output;
}

void FusionTracker::run() {
  DetectionMessage message;
  while (detection_queue_.pop(message)) {
    ingest(message);
  }
}

void FusionTracker::ingest(const DetectionMessage& message) {
  std::lock_guard<std::mutex> lock(mutex_);
  // Track the newest sequence and timestamp so publishers can expose the freshest known state.
  latest_sequence_id_ = std::max(latest_sequence_id_, message.frame_id);
  latest_timestamp_ = std::max(latest_timestamp_, message.timestamp);

  for (const PersonReport& report : message.person) {
    auto best_it = tracks_.end();
    double best_distance = std::numeric_limits<double>::max();

    // Associate by nearest angular distance inside a simple gating window.
    for (auto it = tracks_.begin(); it != tracks_.end(); ++it) {
      const double distance = std::abs(normalize_angle(report.angle - it->angle));
      if (distance < association_gate_degrees_ && distance < best_distance) {
        best_distance = distance;
        best_it = it;
      }
    }

    if (best_it == tracks_.end()) {
      // No compatible track exists yet, so start a new one.
      tracks_.push_back(TrackState{
          .track_id = fmt::format("track-{}", next_track_id_++),
          .angle = report.angle,
          .angle_velocity = 0.0,
          .confidence = 1.0,
          .sources = {report.camera},
          .last_update = message.timestamp,
      });
      continue;
    }

    // Smooth the angle update and estimate angular velocity from time-separated observations.
    const double delta_time_seconds = std::max(
        1e-3, std::chrono::duration<double>(message.timestamp - best_it->last_update).count());
    const double innovation = normalize_angle(report.angle - best_it->angle);
    const double updated_angle = normalize_angle(best_it->angle + (innovation * smoothing_gain_));

    best_it->angle_velocity = innovation / delta_time_seconds;
    best_it->angle = updated_angle;
    best_it->confidence = std::min(1.0, best_it->confidence + 0.05);
    if (!contains_camera(best_it->sources, report.camera)) {
      best_it->sources.push_back(report.camera);
    }
    best_it->last_update = message.timestamp;
  }
}

} // namespace op3
