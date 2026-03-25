#include "core/blocking_queue.hpp"
#include "perception/angle_utils.hpp"
#include "perception/camera_worker.hpp"
#include "perception/detector.hpp"
#include "perception/fusion_tracker.hpp"
#include "perception/json_output.hpp"
#include "perception/mock_person_detector.hpp"
#include "perception/state_publisher.hpp"
#include "perception/types.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>

namespace {

/** Fixed detector used by tests to inject deterministic detections. */
class FixedDetector final : public op3::Detector {
 public:
  FixedDetector(std::vector<op3::Detection> detections, int* call_count)
      : detections_(std::move(detections)), call_count_(call_count) {}

  std::vector<op3::Detection> detect(const cv::Mat&) override {
    if (call_count_ != nullptr) {
      ++(*call_count_);
    }
    return detections_;
  }

 private:
  std::vector<op3::Detection> detections_;
  int* call_count_;
};

TEST(CameraPositionToStringTest, ConvertsAllKnownPositions) {
  EXPECT_EQ(op3::camera_position_to_string(op3::CameraPosition::kLeftFront),
            "left_front");
  EXPECT_EQ(op3::camera_position_to_string(op3::CameraPosition::kRightFront),
            "right_front");
  EXPECT_EQ(op3::camera_position_to_string(op3::CameraPosition::kLeftRear),
            "left_rear");
  EXPECT_EQ(op3::camera_position_to_string(op3::CameraPosition::kRightRear),
            "right_rear");
}

TEST(ComputePersonAngleTest, UsesCameraBaseAngleAtImageCenter) {
  const double angle = op3::compute_person_angle(
      op3::CameraPosition::kLeftFront, cv::Rect(450, 100, 60, 200), 960);

  EXPECT_NEAR(angle, 45.0, 1e-6);
}

TEST(ComputePersonAngleTest, AppliesHorizontalOffsetAtImageEdge) {
  const double angle = op3::compute_person_angle(
      op3::CameraPosition::kRightFront, cv::Rect(0, 100, 60, 200), 960);

  EXPECT_NEAR(angle, -87.1875, 1e-6);
}

TEST(ComputePersonAngleTest, NormalizesRearCameraAnglesToMinus180To180Range) {
  const double angle = op3::compute_person_angle(
      op3::CameraPosition::kLeftRear, cv::Rect(900, 100, 60, 200), 960);

  EXPECT_NEAR(angle, 177.1875, 1e-6);
}

TEST(NormalizeAngleTest, WrapsPastPositiveBoundary) {
  EXPECT_NEAR(op3::normalize_angle(225.0), -135.0, 1e-6);
}

TEST(MockPersonDetectorTest, ReturnsSingleDetectionForDarkBlob) {
  cv::Mat image(300, 400, CV_8UC3, cv::Scalar(235, 235, 235));
  cv::rectangle(image, cv::Rect(120, 80, 50, 120), cv::Scalar(20, 20, 20), cv::FILLED);

  op3::MockPersonDetector detector;
  const std::vector<op3::Detection> detections = detector.detect(image);

  ASSERT_EQ(detections.size(), 1U);
  EXPECT_EQ(detections.front().id, "1");
  EXPECT_EQ(detections.front().bbox, cv::Rect(120, 80, 50, 120));
}

TEST(MockPersonDetectorTest, IgnoresEmptyImage) {
  op3::MockPersonDetector detector;
  const std::vector<op3::Detection> detections = detector.detect(cv::Mat{});

  EXPECT_TRUE(detections.empty());
}

TEST(BlockingQueueTest, DropsOldFramesWhenCapacityIsOne) {
  op3::BlockingQueue<int> queue(1);
  queue.push(1);
  queue.push(2);
  queue.close();

  int value = 0;
  ASSERT_TRUE(queue.pop(value));
  EXPECT_EQ(value, 2);
  EXPECT_FALSE(queue.pop(value));
}

TEST(CameraWorkerTest, EmitsDetectionMessageWithTimestampAndFrameId) {
  op3::BlockingQueue<op3::FrameMessage> input_queue(1);
  op3::BlockingQueue<op3::DetectionMessage> detection_queue(4);

  int call_count = 0;
  op3::CameraWorker worker(
      op3::CameraPosition::kLeftFront,
      std::make_unique<FixedDetector>(
          std::vector<op3::Detection>{
              op3::Detection{.id = "cam", .bbox = cv::Rect(80, 20, 40, 100),
                                    .confidence = 0.9F},
          },
          &call_count),
      input_queue, detection_queue);
  worker.start();

  const auto timestamp = std::chrono::steady_clock::now();
  input_queue.push(op3::FrameMessage{
      .camera = op3::CameraPosition::kLeftFront,
      .frame_id = 11,
      .timestamp = timestamp,
      .image = cv::Mat(200, 400, CV_8UC3, cv::Scalar(235, 235, 235)),
  });
  input_queue.close();
  worker.join();

  detection_queue.close();
  op3::DetectionMessage message{};
  ASSERT_TRUE(detection_queue.pop(message));
  EXPECT_EQ(call_count, 1);
  EXPECT_EQ(message.frame_id, 11U);
  EXPECT_EQ(message.timestamp, timestamp);
  ASSERT_EQ(message.person.size(), 1U);
  EXPECT_EQ(message.person.front().camera, op3::CameraPosition::kLeftFront);
}

TEST(FusionTrackerTest, AssociatesAsynchronousDetectionsIntoSingleTrack) {
  op3::BlockingQueue<op3::DetectionMessage> detection_queue(8);
  op3::FusionTracker tracker(detection_queue, 25.0, 0.5);
  tracker.start();

  const auto base_time = std::chrono::steady_clock::now();
  detection_queue.push(op3::DetectionMessage{
      .camera = op3::CameraPosition::kLeftFront,
      .frame_id = 1,
      .timestamp = base_time,
      .person = {op3::PersonReport{.id = "1", .angle = 30.0,
                                   .camera = op3::CameraPosition::kLeftFront}},
  });
  detection_queue.push(op3::DetectionMessage{
      .camera = op3::CameraPosition::kRightFront,
      .frame_id = 2,
      .timestamp = base_time + std::chrono::milliseconds(15),
      .person = {op3::PersonReport{.id = "1", .angle = 34.0,
                                   .camera = op3::CameraPosition::kRightFront}},
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  tracker.stop();
  tracker.join();

  const op3::PipelineOutput output = tracker.snapshot();
  EXPECT_EQ(output.sequence_id, 2U);
  ASSERT_EQ(output.person.size(), 1U);
  EXPECT_EQ(output.person.front().track_id, "track-1");
  EXPECT_NEAR(output.person.front().angle, 32.0, 1e-6);
  EXPECT_GT(output.person.front().angle_velocity, 0.0);
  std::vector<op3::CameraPosition> sources = output.person.front().sources;
  std::sort(sources.begin(), sources.end(),
            [](op3::CameraPosition lhs, op3::CameraPosition rhs) {
              return static_cast<int>(lhs) < static_cast<int>(rhs);
            });
  ASSERT_EQ(sources.size(), 2U);
  EXPECT_EQ(sources[0], op3::CameraPosition::kLeftFront);
  EXPECT_EQ(sources[1], op3::CameraPosition::kRightFront);
}

TEST(StatePublisherTest, PublishesTrackerSnapshots) {
  op3::BlockingQueue<op3::DetectionMessage> detection_queue(8);
  op3::FusionTracker tracker(detection_queue);
  tracker.start();

  std::atomic<int> publish_count = 0;
  op3::StatePublisher publisher(
      tracker, std::chrono::milliseconds(10),
      [&publish_count](const op3::PipelineOutput& output) {
        if (output.sequence_id > 0) {
          ++publish_count;
        }
      });
  publisher.start();

  detection_queue.push(op3::DetectionMessage{
      .camera = op3::CameraPosition::kLeftFront,
      .frame_id = 5,
      .timestamp = std::chrono::steady_clock::now(),
      .person = {op3::PersonReport{.id = "1", .angle = 30.0,
                                   .camera = op3::CameraPosition::kLeftFront}},
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  tracker.stop();
  tracker.join();
  publisher.stop();
  publisher.join();

  EXPECT_GE(publish_count.load(), 1);
  ASSERT_FALSE(publisher.outputs().empty());
  EXPECT_GE(publisher.outputs().back().sequence_id, 5U);
}

TEST(ToJsonTest, SerializesEmptyOutput) {
  const std::string json = op3::to_json(op3::PipelineOutput{
      .sequence_id = 3,
      .timestamp_ms = 100,
      .person = {},
  });

  EXPECT_EQ(json, "{\"sequence_id\":3,\"timestamp_ms\":100,\"person\":[]}");
}

TEST(ToJsonTest, SerializesTrackedPersonState) {
  const auto timestamp = std::chrono::steady_clock::time_point(std::chrono::milliseconds(250));
  const op3::PipelineOutput output{
      .sequence_id = 9,
      .timestamp_ms = 300,
      .person = {op3::TrackedPerson{
          .track_id = "track-1",
          .angle = 30.0,
          .angle_velocity = 1.5,
          .confidence = 0.95,
          .sources = {op3::CameraPosition::kLeftFront},
          .last_update = timestamp,
      }},
  };

  const std::string json = op3::to_json(output);

  EXPECT_EQ(
      json,
      "{\"sequence_id\":9,\"timestamp_ms\":300,\"person\":[{\"track_id\":\"track-1\",\"angle\":30.0,"
      "\"angle_velocity\":1.5,\"confidence\":0.95,\"sources\":[\"left_front\"],\"last_update_ms\":250}]}");
}

}  // namespace
