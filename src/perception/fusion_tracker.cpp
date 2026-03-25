#include "perception/fusion_tracker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include <fmt/format.h>

#include "perception/angle_utils.hpp"

namespace op3 {

namespace {

constexpr auto kStaleTrackTimeout = std::chrono::milliseconds(250);
constexpr double kInitialConfidence = 0.6;
constexpr double kConfidenceIncrease = 0.1;
constexpr double kConfidenceDecayPerSecond = 0.75;
constexpr double kInitialAngleVariance = 16.0;
constexpr double kInitialVelocityVariance = 900.0;
constexpr double kMeasurementVariance = 4.0;
constexpr double kProcessNoiseIntensity = 400.0;
constexpr double kAssociationEpsilon = 1e-6;

/**
 * Checks whether a track already records a given source camera.
 */
bool contains_camera(const std::vector<CameraPosition>& cameras, CameraPosition camera) {
  return std::find(cameras.begin(), cameras.end(), camera) != cameras.end();
}

/**
 * Clamps confidence to the normalized [0, 1] range.
 */
double clamp_confidence(double confidence) {
  return std::clamp(confidence, 0.0, 1.0);
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

    // Associate by nearest predicted angle inside the gating window. One track can absorb only one
    // observation from the same message.
    for (std::size_t index = 0; index < tracks_.size(); ++index) {
      if (track_assigned[index]) {
        continue;
      }

      const double distance = std::abs(normalize_angle(report.angle - tracks_[index].angle));
      const bool is_better_match = distance + kAssociationEpsilon < best_distance;
      const bool is_tie_break_winner = std::abs(distance - best_distance) <= kAssociationEpsilon &&
                                       best_index < tracks_.size() &&
                                       tracks_[index].last_update > tracks_[best_index].last_update;
      if (distance < association_gate_degrees_ && (is_better_match || is_tie_break_winner)) {
        best_distance = distance;
        best_index = index;
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
  }

  prune_stale_tracks(message.timestamp);
}

void FusionTracker::predict_track(TrackState& track,
                                  std::chrono::steady_clock::time_point timestamp) const {
  const auto delta = timestamp - track.state_timestamp;
  const double delta_seconds = std::chrono::duration<double>(delta).count();
  if (delta_seconds <= 0.0) {
    track.time_since_update = timestamp - track.last_update;
    return;
  }

  const double prior_p00 = track.covariance_00;
  const double prior_p01 = track.covariance_01;
  const double prior_p10 = track.covariance_10;
  const double prior_p11 = track.covariance_11;
  const double delta_squared = delta_seconds * delta_seconds;
  const double delta_cubed = delta_squared * delta_seconds;
  const double delta_fourth = delta_squared * delta_squared;

  const double process_q00 = kProcessNoiseIntensity * delta_fourth * 0.25;
  const double process_q01 = kProcessNoiseIntensity * delta_cubed * 0.5;
  const double process_q11 = kProcessNoiseIntensity * delta_squared;

  track.angle = normalize_angle(track.angle + (track.angle_velocity * delta_seconds));
  track.covariance_00 = prior_p00 + (delta_seconds * (prior_p01 + prior_p10)) +
                        (delta_squared * prior_p11) + process_q00;
  track.covariance_01 = prior_p01 + (delta_seconds * prior_p11) + process_q01;
  track.covariance_10 = prior_p10 + (delta_seconds * prior_p11) + process_q01;
  track.covariance_11 = prior_p11 + process_q11;
  track.state_timestamp = timestamp;
  track.time_since_update = timestamp - track.last_update;
}

void FusionTracker::update_track(TrackState& track, const PersonReport& report,
                                 std::chrono::steady_clock::time_point timestamp) const {
  const double innovation = normalize_angle(report.angle - track.angle);
  const double effective_measurement_variance =
      kMeasurementVariance / std::clamp(smoothing_gain_, 0.1, 1.0);
  const double innovation_covariance = track.covariance_00 + effective_measurement_variance;
  const double kalman_gain_angle = track.covariance_00 / innovation_covariance;
  const double kalman_gain_velocity = track.covariance_10 / innovation_covariance;

  const double prior_p00 = track.covariance_00;
  const double prior_p01 = track.covariance_01;
  const double prior_p10 = track.covariance_10;
  const double prior_p11 = track.covariance_11;

  track.angle = normalize_angle(track.angle + (kalman_gain_angle * innovation));
  track.angle_velocity += kalman_gain_velocity * innovation;

  const double updated_p00 = (1.0 - kalman_gain_angle) * prior_p00;
  const double updated_p01 = (1.0 - kalman_gain_angle) * prior_p01;
  const double updated_p10 = prior_p10 - (kalman_gain_velocity * prior_p00);
  const double updated_p11 = prior_p11 - (kalman_gain_velocity * prior_p01);
  const double symmetric_cross = (updated_p01 + updated_p10) * 0.5;

  track.covariance_00 = updated_p00;
  track.covariance_01 = symmetric_cross;
  track.covariance_10 = symmetric_cross;
  track.covariance_11 = updated_p11;
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
      .angle = normalize_angle(report.angle),
      .angle_velocity = 0.0,
      .covariance_00 = kInitialAngleVariance,
      .covariance_01 = 0.0,
      .covariance_10 = 0.0,
      .covariance_11 = kInitialVelocityVariance,
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
                                 return (timestamp - track.last_update) > kStaleTrackTimeout;
                               }),
                tracks_.end());
}

} // namespace op3
