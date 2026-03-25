#pragma once

#include <chrono>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "perception/fusion_tracker.hpp"

namespace op3 {

/**
 * Periodically snapshots the tracker and forwards stable JSON-ready state updates.
 */
class StatePublisher {
public:
  /**
   * Callback invoked for each published tracker snapshot.
   */
  using OutputCallback = std::function<void(const PipelineOutput&)>;

  /**
   * Creates a publisher that polls tracker state at a fixed interval.
   */
  StatePublisher(FusionTracker& tracker, std::chrono::milliseconds publish_interval,
                 OutputCallback on_output);

  StatePublisher(const StatePublisher&) = delete;
  StatePublisher& operator=(const StatePublisher&) = delete;

  /**
   * Starts the publisher thread.
   */
  void start();

  /**
   * Requests that the publisher stop after the next polling cycle.
   */
  void stop();

  /**
   * Joins the publisher thread.
   */
  void join();

  /**
   * Returns the snapshots emitted so far.
   */
  const std::vector<PipelineOutput>& outputs() const;

private:
  /**
   * Polls the tracker and emits only new sequence ids.
   */
  void run();

  FusionTracker& tracker_;
  std::chrono::milliseconds publish_interval_;
  OutputCallback on_output_;
  mutable std::mutex mutex_;
  std::vector<PipelineOutput> outputs_;
  bool stop_requested_ = false;
  std::thread thread_;
};

} // namespace op3
