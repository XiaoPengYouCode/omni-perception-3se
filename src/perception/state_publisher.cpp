#include "perception/state_publisher.hpp"

#include <thread>
#include <utility>

namespace op3 {

StatePublisher::StatePublisher(FusionTracker& tracker, std::chrono::milliseconds publish_interval,
                               OutputCallback on_output)
    : tracker_(tracker), publish_interval_(publish_interval), on_output_(std::move(on_output)) {}

void StatePublisher::start() {
  thread_ = std::thread([this] { run(); });
}

void StatePublisher::stop() {
  std::lock_guard<std::mutex> lock(mutex_);
  stop_requested_ = true;
}

void StatePublisher::join() {
  if (thread_.joinable()) {
    thread_.join();
  }
}

const std::vector<PipelineOutput>& StatePublisher::outputs() const {
  return outputs_;
}

void StatePublisher::run() {
  std::uint64_t last_published_sequence = 0;

  while (true) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stop_requested_) {
        break;
      }
    }

    // Publish only when fusion has advanced to a newer sequence id.
    const PipelineOutput output = tracker_.snapshot();
    if (output.sequence_id > 0 && output.sequence_id != last_published_sequence) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        outputs_.push_back(output);
      }
      if (on_output_ != nullptr) {
        on_output_(output);
      }
      last_published_sequence = output.sequence_id;
    }

    std::this_thread::sleep_for(publish_interval_);
  }

  // Flush one last snapshot on shutdown so callers can observe the final fused state.
  const PipelineOutput output = tracker_.snapshot();
  if (output.sequence_id > 0 && output.sequence_id != last_published_sequence) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      outputs_.push_back(output);
    }
    if (on_output_ != nullptr) {
      on_output_(output);
    }
  }
}

} // namespace op3
