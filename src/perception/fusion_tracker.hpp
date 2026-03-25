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
    double confidence;
    std::vector<CameraPosition> sources;
    std::chrono::steady_clock::time_point last_update;
  };

  /**
   * Drains asynchronous detection messages until shutdown.
   */
  void run();

  /**
   * Applies one detection message to the current set of fused tracks.
   */
  void ingest(const DetectionMessage& message);

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
