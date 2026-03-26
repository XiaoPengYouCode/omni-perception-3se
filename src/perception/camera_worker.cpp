#include "perception/camera_worker.hpp"

#include <utility>

#include "perception/angle_utils.hpp"

namespace op3 {

namespace {

struct ProcessedFrame {
  std::vector<Detection> detections;
  std::vector<PersonReport> reports;
};

/**
 * Converts detector-space output into robot-centric person reports for one frame.
 */
ProcessedFrame process_camera_frame(const CameraModel& camera_model, const FrameMessage& message,
                                    Detector& detector) {
  ProcessedFrame processed;
  if (message.image.empty()) {
    return processed;
  }

  processed.detections = detector.detect(message.image);
  processed.reports.reserve(processed.detections.size());

  // Convert every raw image-space detection into a unified angular observation.
  for (const Detection& detection : processed.detections) {
    if (detection.bbox.width <= 0 || detection.bbox.height <= 0) {
      continue;
    }

    const double angle =
        compute_person_angle(camera_model.camera, detection.bbox, message.image.cols,
                             camera_model.horizontal_fov_degrees);
    const double camera_range_m = estimate_person_range(detection.bbox, camera_model);
    const cv::Point2d camera_relative_point = body_frame_point_from_polar(camera_range_m, angle);
    const cv::Point2d body_point{camera_model.mount_x_m + camera_relative_point.x,
                                 camera_model.mount_y_m + camera_relative_point.y};
    const cv::Point2d world_point =
        body_to_world_point(body_point.x, body_point.y, message.robot_pose);
    const double range_m = body_frame_range_from_point(body_point.x, body_point.y);
    processed.reports.push_back(PersonReport{
        .id = detection.id,
        .label = detection.label,
        .angle = angle,
        .range_m = range_m,
        .x_m = body_point.x,
        .y_m = body_point.y,
        .world_x_m = world_point.x,
        .world_y_m = world_point.y,
        .camera = camera_model.camera,
    });
  }

  return processed;
}

} // namespace

CameraWorker::CameraWorker(CameraPosition camera, std::unique_ptr<Detector> detector,
                           BlockingQueue<FrameMessage>& input_queue,
                           BlockingQueue<DetectionMessage>& detection_queue)
    : CameraWorker(make_default_camera_model(camera), std::move(detector), input_queue,
                   detection_queue) {}

CameraWorker::CameraWorker(CameraModel camera_model, std::unique_ptr<Detector> detector,
                           BlockingQueue<FrameMessage>& input_queue,
                           BlockingQueue<DetectionMessage>& detection_queue,
                           DetectionCallback on_detection)
    : camera_model_(camera_model), detector_(std::move(detector)), input_queue_(input_queue),
      detection_queue_(detection_queue), on_detection_(std::move(on_detection)) {}

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
    if (message.camera != camera_model_.camera || detector_ == nullptr) {
      continue;
    }

    const ProcessedFrame processed = process_camera_frame(camera_model_, message, *detector_);
    if (on_detection_ != nullptr) {
      on_detection_(message, processed.detections, processed.reports);
    }

    // Emit asynchronous detections so fusion can operate independently from camera timing.
    detection_queue_.push(DetectionMessage{
        .camera = camera_model_.camera,
        .frame_id = message.frame_id,
        .timestamp = message.timestamp,
        .robot_pose = message.robot_pose,
        .person = std::move(processed.reports),
    });
  }
}

} // namespace op3
