#pragma once

#include <memory>
#include <thread>

#include "core/blocking_queue.hpp"
#include "perception/detector.hpp"
#include "perception/types.hpp"

namespace op3 {

/**
 * Long-lived worker that performs inference for one camera stream.
 */
class CameraWorker {
 public:
  /**
   * Creates a worker bound to one camera and one detector instance.
   */
  CameraWorker(CameraPosition camera, std::unique_ptr<Detector> detector,
               BlockingQueue<FrameMessage>& input_queue,
               BlockingQueue<DetectionMessage>& detection_queue);

  CameraWorker(const CameraWorker&) = delete;
  CameraWorker& operator=(const CameraWorker&) = delete;

  /**
   * Starts the worker thread.
   */
  void start();

  /**
   * Joins the worker thread if it is still running.
   */
  void join();

 private:
  /**
   * Consumes frame messages, runs detection, and publishes detection messages.
   */
  void run();

  CameraPosition camera_;
  std::unique_ptr<Detector> detector_;
  BlockingQueue<FrameMessage>& input_queue_;
  BlockingQueue<DetectionMessage>& detection_queue_;
  std::thread thread_;
};

}  // namespace op3
