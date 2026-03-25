#include "perception/camera_worker.hpp"

#include "perception/angle_utils.hpp"

namespace op3 {

namespace {

/**
 * Converts detector-space output into robot-centric person reports for one frame.
 */
std::vector<PersonReport> process_camera_frame(CameraPosition camera, const cv::Mat& image,
                                               Detector& detector) {
  std::vector<PersonReport> reports;
  if (image.empty()) {
    return reports;
  }

  const std::vector<Detection> detections = detector.detect(image);
  reports.reserve(detections.size());

  // Convert every raw image-space detection into a unified angular observation.
  for (const Detection& detection : detections) {
    reports.push_back(PersonReport{
        .id = detection.id,
        .angle = compute_person_angle(camera, detection.bbox, image.cols),
        .camera = camera,
    });
  }

  return reports;
}

}  // namespace

CameraWorker::CameraWorker(CameraPosition camera, std::unique_ptr<Detector> detector,
                           BlockingQueue<FrameMessage>& input_queue,
                           BlockingQueue<DetectionMessage>& detection_queue)
    : camera_(camera),
      detector_(std::move(detector)),
      input_queue_(input_queue),
      detection_queue_(detection_queue) {}

void CameraWorker::start() {
  thread_ = std::thread([this] { run(); });
}

void CameraWorker::join() {
  if (thread_.joinable()) {
    thread_.join();
  }
}

void CameraWorker::run() {
  FrameMessage message;
  while (input_queue_.pop(message)) {
    if (message.camera != camera_ || detector_ == nullptr) {
      continue;
    }

    // Emit asynchronous detections so fusion can operate independently from camera timing.
    detection_queue_.push(DetectionMessage{
        .camera = camera_,
        .frame_id = message.frame_id,
        .timestamp = message.timestamp,
        .person = process_camera_frame(camera_, message.image, *detector_),
    });
  }
}

}  // namespace op3
