#pragma once

#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "core/blocking_queue.hpp"
#include "perception/types.hpp"

namespace op3 {

/**
 * Maintains fused person tracks from asynchronous per-camera detections.
 */
class FusionTracker {
public:
  /**
   * Creates a tracker with simple angle-based data association parameters.
   */
  FusionTracker(BlockingQueue<DetectionMessage>& detection_queue,
                double association_gate_degrees = 20.0, double smoothing_gain = 0.35);

  FusionTracker(const FusionTracker&) = delete;
  FusionTracker& operator=(const FusionTracker&) = delete;

  /**
   * Starts the tracker thread.
   */
  void start();

  /**
   * Requests shutdown by closing the input detection queue.
   */
  void stop();

  /**
   * Joins the tracker thread.
   */
  void join();

  /**
   * Returns a thread-safe snapshot of the current fused state.
   */
  PipelineOutput snapshot() const;

private:
  /**
   * Internal mutable track state owned by the fusion thread.
   */
  struct TrackState {
    std::string track_id;
    double angle;
    double angle_velocity;
    double covariance_00;
    double covariance_01;
    double covariance_10;
    double covariance_11;
    double confidence;
    std::vector<CameraPosition> sources;
    std::chrono::steady_clock::time_point state_timestamp;
    std::chrono::steady_clock::time_point last_update;
    std::chrono::steady_clock::duration time_since_update{};
    std::uint64_t hit_count = 0;
    std::uint64_t missed_update_count = 0;
  };

  /**
   * Drains asynchronous detection messages until shutdown.
   */
  void run();

  /**
   * Applies one detection message to the current set of fused tracks.
   */
  void ingest(const DetectionMessage& message);

  /**
   * Propagates a track to the requested timestamp using a constant-velocity model.
   */
  void predict_track(TrackState& track, std::chrono::steady_clock::time_point timestamp) const;

  /**
   * Applies a scalar angle measurement update to a predicted track.
   */
  void update_track(TrackState& track, const PersonReport& report,
                    std::chrono::steady_clock::time_point timestamp) const;

  /**
   * Creates a new track from an unmatched observation.
   */
  void create_track(const PersonReport& report, std::chrono::steady_clock::time_point timestamp);

  /**
   * Removes tracks that have gone stale and decays unmatched tracks.
   */
  void prune_stale_tracks(std::chrono::steady_clock::time_point timestamp);

  BlockingQueue<DetectionMessage>& detection_queue_;
  double association_gate_degrees_;
  double smoothing_gain_;
  mutable std::mutex mutex_;
  std::vector<TrackState> tracks_;
  std::uint64_t next_track_id_ = 1;
  std::uint64_t latest_sequence_id_ = 0;
  std::chrono::steady_clock::time_point latest_timestamp_{};
  std::thread thread_;
};

} // namespace op3
