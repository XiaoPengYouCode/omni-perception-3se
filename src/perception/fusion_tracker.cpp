#include "perception/fusion_tracker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include <fmt/format.h>

#include "perception/angle_utils.hpp"

namespace op3 {

namespace {

constexpr auto kStaleTrackTimeout = std::chrono::milliseconds(1000);
constexpr double kInitialConfidence = 0.55;
constexpr double kConfidenceIncrease = 0.12;
constexpr double kConfidenceDecayPerSecond = 0.3;
constexpr double kInitialRadiusM = 0.9;
constexpr double kMinRadiusM = 0.35;
constexpr double kMaxRadiusM = 3.5;
constexpr double kRadiusGrowthPerSecond = 0.5;
constexpr double kAssociationEpsilon = 1e-6;
constexpr double kPi = 3.14159265358979323846;

bool contains_camera(const std::vector<CameraPosition>& cameras, CameraPosition camera) {
  return std::find(cameras.begin(), cameras.end(), camera) != cameras.end();
}

bool labels_match(const std::string& lhs, const std::string& rhs) {
  return !lhs.empty() && !rhs.empty() && lhs == rhs;
}

double clamp_confidence(double confidence) {
  return std::clamp(confidence, 0.0, 1.0);
}

double clamp_radius(double radius_m) {
  return std::clamp(radius_m, kMinRadiusM, kMaxRadiusM);
}

std::int64_t to_epoch_milliseconds(std::chrono::steady_clock::time_point timestamp) {
  return static_cast<std::int64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(timestamp.time_since_epoch()).count());
}

} // namespace

FusionTracker::FusionTracker(BlockingQueue<DetectionMessage>& detection_queue,
                             double association_gate_m, double position_gain, double velocity_gain)
    : detection_queue_(detection_queue), association_gate_m_(association_gate_m),
      position_gain_(position_gain), velocity_gain_(velocity_gain) {}

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
    const cv::Point2d body_point =
        world_to_body_point(track.world_x_m, track.world_y_m, latest_robot_pose_);
    const double range_m = body_frame_range_from_point(body_point.x, body_point.y);
    const double angle = body_frame_angle_from_point(body_point.x, body_point.y);
    output.person.push_back(TrackedPerson{
        .track_id = track.track_id,
        .label = track.label,
        .x_m = body_point.x,
        .y_m = body_point.y,
        .world_x_m = track.world_x_m,
        .world_y_m = track.world_y_m,
        .vx_mps = track.vx_mps,
        .vy_mps = track.vy_mps,
        .range_m = range_m,
        .radius_m = track.radius_m,
        .angle = angle,
        .angle_velocity = track.vx_mps == 0.0 && track.vy_mps == 0.0
                              ? 0.0
                              : (body_point.x * track.vy_mps - body_point.y * track.vx_mps) /
                                    std::max(range_m * range_m, 1e-6) * (180.0 / kPi),
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
  latest_sequence_id_ = std::max(latest_sequence_id_, message.frame_id);
  latest_timestamp_ = std::max(latest_timestamp_, message.timestamp);
  latest_robot_pose_ = message.robot_pose;

  std::vector<std::chrono::steady_clock::duration> prediction_steps;
  prediction_steps.reserve(tracks_.size());
  for (TrackState& track : tracks_) {
    const auto prediction_step = std::max(std::chrono::steady_clock::duration::zero(),
                                          message.timestamp - track.state_timestamp);
    prediction_steps.push_back(prediction_step);
    predict_track(track, message.timestamp);
  }

  std::vector<bool> track_assigned(tracks_.size(), false);
  for (const PersonReport& report : message.person) {
    std::size_t best_index = tracks_.size();
    double best_distance = std::numeric_limits<double>::max();

    if (!report.label.empty()) {
      for (std::size_t index = 0; index < tracks_.size(); ++index) {
        if (!labels_match(tracks_[index].label, report.label)) {
          continue;
        }
        best_index = index;
        break;
      }
    }

    if (best_index == tracks_.size()) {
      for (std::size_t index = 0; index < tracks_.size(); ++index) {
        if (track_assigned[index]) {
          continue;
        }
        if (!report.label.empty() && !tracks_[index].label.empty() &&
            tracks_[index].label != report.label) {
          continue;
        }

        const double distance = association_distance(tracks_[index], report);
        const bool is_better_match = distance + kAssociationEpsilon < best_distance;
        const bool is_tie_break_winner =
            std::abs(distance - best_distance) <= kAssociationEpsilon &&
            best_index < tracks_.size() &&
            tracks_[index].last_update > tracks_[best_index].last_update;
        if (distance < association_gate_m_ && (is_better_match || is_tie_break_winner)) {
          best_distance = distance;
          best_index = index;
        }
      }
    }

    if (best_index == tracks_.size()) {
      create_track(report, message.timestamp);
      track_assigned.push_back(true);
      prediction_steps.push_back(std::chrono::steady_clock::duration::zero());
      continue;
    }

    update_track(tracks_[best_index], report, message.timestamp);
    track_assigned[best_index] = true;
  }

  for (std::size_t index = 0; index < tracks_.size(); ++index) {
    if (track_assigned[index]) {
      continue;
    }

    tracks_[index].missed_update_count += 1;
    const double missed_seconds = std::chrono::duration<double>(prediction_steps[index]).count();
    tracks_[index].confidence =
        clamp_confidence(tracks_[index].confidence - (missed_seconds * kConfidenceDecayPerSecond));
    tracks_[index].radius_m =
        clamp_radius(tracks_[index].radius_m + (missed_seconds * kRadiusGrowthPerSecond));
  }

  prune_stale_tracks(message.timestamp);
  enforce_unique_labeled_tracks();
}

void FusionTracker::predict_track(TrackState& track,
                                  std::chrono::steady_clock::time_point timestamp) const {
  const auto delta = timestamp - track.state_timestamp;
  const double delta_seconds = std::chrono::duration<double>(delta).count();
  if (delta_seconds <= 0.0) {
    track.time_since_update = timestamp - track.last_update;
    return;
  }

  track.world_x_m += track.vx_mps * delta_seconds;
  track.world_y_m += track.vy_mps * delta_seconds;
  track.radius_m = clamp_radius(track.radius_m + (delta_seconds * kRadiusGrowthPerSecond * 0.4));
  track.state_timestamp = timestamp;
  track.time_since_update = timestamp - track.last_update;
}

void FusionTracker::update_track(TrackState& track, const PersonReport& report,
                                 std::chrono::steady_clock::time_point timestamp) const {
  const double delta_seconds =
      std::max(std::chrono::duration<double>(timestamp - track.last_update).count(), 1e-3);
  const double predicted_x = track.world_x_m;
  const double predicted_y = track.world_y_m;
  const double updated_x = predicted_x + (position_gain_ * (report.world_x_m - predicted_x));
  const double updated_y = predicted_y + (position_gain_ * (report.world_y_m - predicted_y));
  const double measured_vx = (report.world_x_m - predicted_x) / delta_seconds;
  const double measured_vy = (report.world_y_m - predicted_y) / delta_seconds;

  track.world_x_m = updated_x;
  track.world_y_m = updated_y;
  track.vx_mps = ((1.0 - velocity_gain_) * track.vx_mps) + (velocity_gain_ * measured_vx);
  track.vy_mps = ((1.0 - velocity_gain_) * track.vy_mps) + (velocity_gain_ * measured_vy);
  track.radius_m = clamp_radius((track.radius_m * 0.55) +
                                (std::abs(track.range_m - report.range_m) * 0.2) + 0.25);
  track.range_m = report.range_m;
  if (track.label.empty() && !report.label.empty()) {
    track.label = report.label;
  }
  track.confidence = clamp_confidence(track.confidence + kConfidenceIncrease);
  track.state_timestamp = timestamp;
  track.last_update = timestamp;
  track.time_since_update = std::chrono::steady_clock::duration::zero();
  track.hit_count += 1;
  track.missed_update_count = 0;
  if (!contains_camera(track.sources, report.camera)) {
    track.sources.push_back(report.camera);
  }
}

void FusionTracker::create_track(const PersonReport& report,
                                 std::chrono::steady_clock::time_point timestamp) {
  tracks_.push_back(TrackState{
      .track_id = fmt::format("track-{}", next_track_id_++),
      .label = report.label,
      .world_x_m = report.world_x_m,
      .world_y_m = report.world_y_m,
      .vx_mps = 0.0,
      .vy_mps = 0.0,
      .range_m = report.range_m,
      .radius_m = kInitialRadiusM,
      .confidence = kInitialConfidence,
      .sources = {report.camera},
      .state_timestamp = timestamp,
      .last_update = timestamp,
      .time_since_update = std::chrono::steady_clock::duration::zero(),
      .hit_count = 1,
      .missed_update_count = 0,
  });
}

void FusionTracker::prune_stale_tracks(std::chrono::steady_clock::time_point timestamp) {
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                               [timestamp](const TrackState& track) {
                                 return (timestamp - track.last_update) > kStaleTrackTimeout ||
                                        track.confidence <= 0.01;
                               }),
                tracks_.end());
}

void FusionTracker::enforce_unique_labeled_tracks() {
  auto is_better_track = [](const TrackState& lhs, const TrackState& rhs) {
    if (lhs.last_update != rhs.last_update) {
      return lhs.last_update > rhs.last_update;
    }
    if (std::abs(lhs.confidence - rhs.confidence) > kAssociationEpsilon) {
      return lhs.confidence > rhs.confidence;
    }
    if (lhs.hit_count != rhs.hit_count) {
      return lhs.hit_count > rhs.hit_count;
    }
    return lhs.track_id < rhs.track_id;
  };

  std::vector<TrackState> unique_tracks;
  unique_tracks.reserve(tracks_.size());
  for (const TrackState& track : tracks_) {
    if (track.label.empty()) {
      unique_tracks.push_back(track);
      continue;
    }

    auto existing = std::find_if(unique_tracks.begin(), unique_tracks.end(),
                                 [&track](const TrackState& candidate) {
                                   return labels_match(candidate.label, track.label);
                                 });
    if (existing == unique_tracks.end()) {
      unique_tracks.push_back(track);
      continue;
    }

    TrackState merged = is_better_track(track, *existing) ? track : *existing;
    const TrackState& other = is_better_track(track, *existing) ? *existing : track;
    for (CameraPosition source : other.sources) {
      if (!contains_camera(merged.sources, source)) {
        merged.sources.push_back(source);
      }
    }
    merged.confidence = std::max(merged.confidence, other.confidence);
    merged.radius_m = std::min(merged.radius_m, other.radius_m);
    *existing = merged;
  }

  tracks_ = std::move(unique_tracks);
}

double FusionTracker::association_distance(const TrackState& track,
                                           const PersonReport& report) const {
  return std::hypot(report.world_x_m - track.world_x_m, report.world_y_m - track.world_y_m);
}

} // namespace op3
