#include "core/blocking_queue.hpp"
#include "perception/camera_worker.hpp"
#include "perception/demo_frames.hpp"
#include "perception/fusion_tracker.hpp"
#include "perception/json_output.hpp"
#include "perception/mock_person_detector.hpp"
#include "perception/state_publisher.hpp"
#include "perception/types.hpp"

#include <chrono>
#include <cstdlib>
#include <thread>

#include <fmt/format.h>
#include <opencv2/core/utils/logger.hpp>

int main() {
  // Keep OpenCV quiet so stdout stays machine-readable JSON.
  setenv("OPENCV_LOG_LEVEL", "ERROR", 1);
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

  // Build a fully synthetic four-camera scene so the pipeline can run without hardware.
  const std::vector<op3::FrameInput> frames = op3::make_demo_frames();
  op3::BlockingQueue<op3::FrameMessage> left_front_queue(1);
  op3::BlockingQueue<op3::FrameMessage> right_front_queue(1);
  op3::BlockingQueue<op3::FrameMessage> left_rear_queue(1);
  op3::BlockingQueue<op3::FrameMessage> right_rear_queue(1);
  op3::BlockingQueue<op3::DetectionMessage> detection_queue(32);

  // The tracker fuses asynchronous detections while the publisher exposes state snapshots.
  op3::FusionTracker tracker(detection_queue);
  tracker.start();

  op3::StatePublisher publisher(
      tracker, std::chrono::milliseconds(20),
      [](const op3::PipelineOutput& output) {
        if (output.sequence_id == 0) {
          return;
        }
        fmt::print("{}\n", op3::to_json(output));
      });
  publisher.start();

  // Each camera owns its own detector instance and worker thread.
  op3::CameraWorker left_front_worker(
      op3::CameraPosition::kLeftFront, std::make_unique<op3::MockPersonDetector>(),
      left_front_queue, detection_queue);
  op3::CameraWorker right_front_worker(
      op3::CameraPosition::kRightFront, std::make_unique<op3::MockPersonDetector>(),
      right_front_queue, detection_queue);
  op3::CameraWorker left_rear_worker(
      op3::CameraPosition::kLeftRear, std::make_unique<op3::MockPersonDetector>(),
      left_rear_queue, detection_queue);
  op3::CameraWorker right_rear_worker(
      op3::CameraPosition::kRightRear, std::make_unique<op3::MockPersonDetector>(),
      right_rear_queue, detection_queue);

  left_front_worker.start();
  right_front_worker.start();
  left_rear_worker.start();
  right_rear_worker.start();

  std::uint64_t frame_id = 1;
  auto timestamp = std::chrono::steady_clock::now();
  for (const op3::FrameInput& frame : frames) {
    const op3::FrameMessage message{
        .camera = frame.camera,
        .frame_id = frame_id++,
        .timestamp = timestamp,
        .image = frame.image,
    };

    // Feed each camera queue independently to mimic unsynchronized camera arrivals.
    switch (frame.camera) {
      case op3::CameraPosition::kLeftFront:
        left_front_queue.push(message);
        break;
      case op3::CameraPosition::kRightFront:
        right_front_queue.push(message);
        break;
      case op3::CameraPosition::kLeftRear:
        left_rear_queue.push(message);
        break;
      case op3::CameraPosition::kRightRear:
        right_rear_queue.push(message);
        break;
    }

    timestamp += std::chrono::milliseconds(5);
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(60));

  // Shut workers down first, then stop fusion and publishing after input has drained.
  left_front_queue.close();
  right_front_queue.close();
  left_rear_queue.close();
  right_rear_queue.close();

  left_front_worker.join();
  right_front_worker.join();
  left_rear_worker.join();
  right_rear_worker.join();

  tracker.stop();
  tracker.join();

  publisher.stop();
  publisher.join();

  return 0;
}
