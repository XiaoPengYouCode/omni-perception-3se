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
constexpr double kBaseProcessNoiseM = 0.3;
constexpr double kBaseVelocityNoiseMps = 1.2;
constexpr double kBaseMeasurementNoiseM = 0.25;

using StateVector = std::array<double, 4>;
using Matrix4 = std::array<double, 16>;
using Matrix2 = std::array<double, 4>;
using Matrix2x4 = std::array<double, 8>;
using Matrix4x2 = std::array<double, 8>;

double state_x(const StateVector& state) {
  return state[0];
}

double state_y(const StateVector& state) {
  return state[1];
}

double state_vx(const StateVector& state) {
  return state[2];
}

double state_vy(const StateVector& state) {
  return state[3];
}

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

Matrix4 identity4() {
  return Matrix4{
      1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
  };
}

Matrix4 transpose4(const Matrix4& matrix) {
  return Matrix4{
      matrix[0], matrix[4], matrix[8],  matrix[12], matrix[1], matrix[5], matrix[9],  matrix[13],
      matrix[2], matrix[6], matrix[10], matrix[14], matrix[3], matrix[7], matrix[11], matrix[15],
  };
}

Matrix4 add4(const Matrix4& lhs, const Matrix4& rhs) {
  Matrix4 result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = lhs[index] + rhs[index];
  }
  return result;
}

Matrix2 add2(const Matrix2& lhs, const Matrix2& rhs) {
  Matrix2 result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = lhs[index] + rhs[index];
  }
  return result;
}

Matrix4 subtract4(const Matrix4& lhs, const Matrix4& rhs) {
  Matrix4 result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = lhs[index] - rhs[index];
  }
  return result;
}

StateVector multiply4x4_vector(const Matrix4& matrix, const StateVector& vector) {
  return StateVector{
      (matrix[0] * vector[0]) + (matrix[1] * vector[1]) + (matrix[2] * vector[2]) +
          (matrix[3] * vector[3]),
      (matrix[4] * vector[0]) + (matrix[5] * vector[1]) + (matrix[6] * vector[2]) +
          (matrix[7] * vector[3]),
      (matrix[8] * vector[0]) + (matrix[9] * vector[1]) + (matrix[10] * vector[2]) +
          (matrix[11] * vector[3]),
      (matrix[12] * vector[0]) + (matrix[13] * vector[1]) + (matrix[14] * vector[2]) +
          (matrix[15] * vector[3]),
  };
}

Matrix4 multiply4x4(const Matrix4& lhs, const Matrix4& rhs) {
  Matrix4 result{};
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      double value = 0.0;
      for (int inner = 0; inner < 4; ++inner) {
        value += lhs[(row * 4) + inner] * rhs[(inner * 4) + col];
      }
      result[(row * 4) + col] = value;
    }
  }
  return result;
}

Matrix4x2 multiply4x4_4x2(const Matrix4& lhs, const Matrix4x2& rhs) {
  Matrix4x2 result{};
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 2; ++col) {
      double value = 0.0;
      for (int inner = 0; inner < 4; ++inner) {
        value += lhs[(row * 4) + inner] * rhs[(inner * 2) + col];
      }
      result[(row * 2) + col] = value;
    }
  }
  return result;
}

Matrix2x4 multiply2x4_4x4(const Matrix2x4& lhs, const Matrix4& rhs) {
  Matrix2x4 result{};
  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < 4; ++col) {
      double value = 0.0;
      for (int inner = 0; inner < 4; ++inner) {
        value += lhs[(row * 4) + inner] * rhs[(inner * 4) + col];
      }
      result[(row * 4) + col] = value;
    }
  }
  return result;
}

Matrix2 multiply2x4_4x2(const Matrix2x4& lhs, const Matrix4x2& rhs) {
  Matrix2 result{};
  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < 2; ++col) {
      double value = 0.0;
      for (int inner = 0; inner < 4; ++inner) {
        value += lhs[(row * 4) + inner] * rhs[(inner * 2) + col];
      }
      result[(row * 2) + col] = value;
    }
  }
  return result;
}

Matrix4 multiply4x2_2x4(const Matrix4x2& lhs, const Matrix2x4& rhs) {
  Matrix4 result{};
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      double value = 0.0;
      for (int inner = 0; inner < 2; ++inner) {
        value += lhs[(row * 2) + inner] * rhs[(inner * 4) + col];
      }
      result[(row * 4) + col] = value;
    }
  }
  return result;
}

StateVector multiply4x2_vector(const Matrix4x2& matrix, const std::array<double, 2>& vector) {
  return StateVector{
      (matrix[0] * vector[0]) + (matrix[1] * vector[1]),
      (matrix[2] * vector[0]) + (matrix[3] * vector[1]),
      (matrix[4] * vector[0]) + (matrix[5] * vector[1]),
      (matrix[6] * vector[0]) + (matrix[7] * vector[1]),
  };
}

Matrix4x2 transpose2x4(const Matrix2x4& matrix) {
  return Matrix4x2{
      matrix[0], matrix[4], matrix[1], matrix[5], matrix[2], matrix[6], matrix[3], matrix[7],
  };
}

Matrix2x4 transpose4x2(const Matrix4x2& matrix) {
  return Matrix2x4{
      matrix[0], matrix[2], matrix[4], matrix[6], matrix[1], matrix[3], matrix[5], matrix[7],
  };
}

Matrix2 invert2x2(const Matrix2& matrix) {
  const double determinant = (matrix[0] * matrix[3]) - (matrix[1] * matrix[2]);
  if (std::abs(determinant) <= 1e-9) {
    return Matrix2{
        1e6,
        0.0,
        0.0,
        1e6,
    };
  }

  const double inverse_determinant = 1.0 / determinant;
  return Matrix2{
      matrix[3] * inverse_determinant,
      -matrix[1] * inverse_determinant,
      -matrix[2] * inverse_determinant,
      matrix[0] * inverse_determinant,
  };
}

Matrix4x2 multiply4x2_2x2(const Matrix4x2& lhs, const Matrix2& rhs) {
  Matrix4x2 result{};
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 2; ++col) {
      double value = 0.0;
      for (int inner = 0; inner < 2; ++inner) {
        value += lhs[(row * 2) + inner] * rhs[(inner * 2) + col];
      }
      result[(row * 2) + col] = value;
    }
  }
  return result;
}

Matrix4 make_transition_matrix(double delta_seconds) {
  return Matrix4{
      1.0, 0.0, delta_seconds, 0.0, 0.0, 1.0, 0.0, delta_seconds,
      0.0, 0.0, 1.0,           0.0, 0.0, 0.0, 0.0, 1.0,
  };
}

Matrix4 make_process_noise(double delta_seconds, double position_gain, double velocity_gain) {
  const double dt2 = delta_seconds * delta_seconds;
  const double dt3 = dt2 * delta_seconds;
  const double dt4 = dt2 * dt2;
  const double position_variance =
      std::max(0.05, kBaseProcessNoiseM * (1.0 + (1.0 - position_gain)));
  const double velocity_variance =
      std::max(0.1, kBaseVelocityNoiseMps * (1.0 + (1.0 - velocity_gain)));
  const double acceleration_variance =
      std::max(position_variance * position_variance, velocity_variance * velocity_variance);

  return Matrix4{
      0.25 * dt4 * acceleration_variance,
      0.0,
      0.5 * dt3 * acceleration_variance,
      0.0,
      0.0,
      0.25 * dt4 * acceleration_variance,
      0.0,
      0.5 * dt3 * acceleration_variance,
      0.5 * dt3 * acceleration_variance,
      0.0,
      dt2 * acceleration_variance,
      0.0,
      0.0,
      0.5 * dt3 * acceleration_variance,
      0.0,
      dt2 * acceleration_variance,
  };
}

Matrix2 make_measurement_noise(double range_m, double position_gain) {
  const double sigma =
      std::max(0.2, kBaseMeasurementNoiseM * (1.0 + (1.0 - position_gain)) + (range_m * 0.03));
  const double variance = sigma * sigma;
  return Matrix2{
      variance,
      0.0,
      0.0,
      variance,
  };
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
        world_to_body_point(state_x(track.state), state_y(track.state), latest_robot_pose_);
    const double range_m = body_frame_range_from_point(body_point.x, body_point.y);
    const double angle = body_frame_angle_from_point(body_point.x, body_point.y);
    output.person.push_back(TrackedPerson{
        .track_id = track.track_id,
        .label = track.label,
        .x_m = body_point.x,
        .y_m = body_point.y,
        .world_x_m = state_x(track.state),
        .world_y_m = state_y(track.state),
        .vx_mps = state_vx(track.state),
        .vy_mps = state_vy(track.state),
        .range_m = range_m,
        .radius_m = track.radius_m,
        .angle = angle,
        .angle_velocity =
            state_vx(track.state) == 0.0 && state_vy(track.state) == 0.0
                ? 0.0
                : (body_point.x * state_vy(track.state) - body_point.y * state_vx(track.state)) /
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

  const Matrix4 transition = make_transition_matrix(delta_seconds);
  track.state = multiply4x4_vector(transition, track.state);
  track.covariance =
      add4(multiply4x4(multiply4x4(transition, track.covariance), transpose4(transition)),
           make_process_noise(delta_seconds, position_gain_, velocity_gain_));
  track.radius_m = clamp_radius(track.radius_m + (delta_seconds * kRadiusGrowthPerSecond * 0.4));
  track.state_timestamp = timestamp;
  track.time_since_update = timestamp - track.last_update;
}

void FusionTracker::update_track(TrackState& track, const PersonReport& report,
                                 std::chrono::steady_clock::time_point timestamp) const {
  const Matrix2x4 measurement_jacobian = {
      1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
  };
  const Matrix4x2 measurement_jacobian_transpose = transpose2x4(measurement_jacobian);
  const std::array<double, 2> measurement = {report.world_x_m, report.world_y_m};
  const std::array<double, 2> predicted_measurement = {track.state[0], track.state[1]};
  const std::array<double, 2> residual = {
      measurement[0] - predicted_measurement[0],
      measurement[1] - predicted_measurement[1],
  };
  const Matrix2 measurement_noise = make_measurement_noise(report.range_m, position_gain_);

  const Matrix2 innovation_covariance =
      add2(multiply2x4_4x2(multiply2x4_4x4(measurement_jacobian, track.covariance),
                           measurement_jacobian_transpose),
           measurement_noise);
  const Matrix4x2 kalman_gain = multiply4x4_4x2(track.covariance, measurement_jacobian_transpose);
  const Matrix2 innovation_inverse = invert2x2(innovation_covariance);

  const Matrix4x2 normalized_kalman_gain = multiply4x2_2x2(kalman_gain, innovation_inverse);

  const StateVector correction = multiply4x2_vector(normalized_kalman_gain, residual);
  for (int index = 0; index < 4; ++index) {
    track.state[index] += correction[index];
  }

  const Matrix4 covariance_update = multiply4x2_2x4(normalized_kalman_gain, measurement_jacobian);
  const Matrix4 residual_projection = subtract4(identity4(), covariance_update);
  const Matrix4 residual_projection_transpose = transpose4(residual_projection);
  const Matrix4x2 gain_times_measurement_noise =
      multiply4x2_2x2(normalized_kalman_gain, measurement_noise);
  track.covariance =
      add4(multiply4x4(multiply4x4(residual_projection, track.covariance),
                       residual_projection_transpose),
           multiply4x2_2x4(gain_times_measurement_noise, transpose4x2(normalized_kalman_gain)));

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
  TrackState track{
      .track_id = fmt::format("track-{}", next_track_id_++),
      .label = report.label,
      .state = {report.world_x_m, report.world_y_m, 0.0, 0.0},
      .covariance =
          {
              0.8,
              0.0,
              0.0,
              0.0,
              0.0,
              0.8,
              0.0,
              0.0,
              0.0,
              0.0,
              25.0,
              0.0,
              0.0,
              0.0,
              0.0,
              25.0,
          },
      .range_m = report.range_m,
      .radius_m = kInitialRadiusM,
      .confidence = kInitialConfidence,
      .sources = {report.camera},
      .state_timestamp = timestamp,
      .last_update = timestamp,
      .time_since_update = std::chrono::steady_clock::duration::zero(),
      .hit_count = 1,
      .missed_update_count = 0,
  };
  tracks_.push_back(track);
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
  return std::hypot(report.world_x_m - state_x(track.state),
                    report.world_y_m - state_y(track.state));
}

} // namespace op3
