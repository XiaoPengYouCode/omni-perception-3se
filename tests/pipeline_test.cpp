#include "core/blocking_queue.hpp"
#include "perception/angle_utils.hpp"
#include "perception/camera_worker.hpp"
#include "perception/detector.hpp"
#include "perception/fusion_tracker.hpp"
#include "perception/json_output.hpp"
#include "perception/mock_person_detector.hpp"
#include "perception/scene_labels.hpp"
#include "perception/state_publisher.hpp"
#include "perception/types.hpp"
#include "simulation/simulation_engine.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <utility>

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

op3::PersonReport make_report(std::string id, double angle_degrees, double range_m,
                              op3::CameraPosition camera, std::string label = {}) {
  const cv::Point2d point = op3::body_frame_point_from_polar(range_m, angle_degrees);
  return op3::PersonReport{
      .id = std::move(id),
      .label = std::move(label),
      .angle = angle_degrees,
      .range_m = range_m,
      .x_m = point.x,
      .y_m = point.y,
      .world_x_m = point.x,
      .world_y_m = point.y,
      .camera = camera,
  };
}

void allow_tracker_to_drain(std::chrono::milliseconds wait = std::chrono::milliseconds(40)) {
  std::this_thread::sleep_for(wait);
}

TEST(CameraPositionToStringTest, ConvertsAllKnownPositions) {
  EXPECT_EQ(op3::camera_position_to_string(op3::CameraPosition::kLeftFront), "left_front");
  EXPECT_EQ(op3::camera_position_to_string(op3::CameraPosition::kRightFront), "right_front");
  EXPECT_EQ(op3::camera_position_to_string(op3::CameraPosition::kLeftRear), "left_rear");
  EXPECT_EQ(op3::camera_position_to_string(op3::CameraPosition::kRightRear), "right_rear");
}

TEST(ComputePersonAngleTest, UsesCameraBaseAngleAtImageCenter) {
  const double angle =
      op3::compute_person_angle(op3::CameraPosition::kLeftFront, cv::Rect(450, 100, 60, 200), 960);

  EXPECT_NEAR(angle, 45.0, 1e-6);
}

TEST(ComputePersonAngleTest, AppliesHorizontalOffsetAtImageEdge) {
  const double angle =
      op3::compute_person_angle(op3::CameraPosition::kRightFront, cv::Rect(0, 100, 60, 200), 960);

  EXPECT_NEAR(angle, -87.1875, 1e-6);
}

TEST(ComputePersonAngleTest, NormalizesRearCameraAnglesToMinus180To180Range) {
  const double angle =
      op3::compute_person_angle(op3::CameraPosition::kLeftRear, cv::Rect(900, 100, 60, 200), 960);

  EXPECT_NEAR(angle, 177.1875, 1e-6);
}

TEST(EstimatePersonRangeTest, LargerBoundingBoxesProduceCloserRanges) {
  const op3::CameraModel model = op3::make_default_camera_model(op3::CameraPosition::kLeftFront);

  const double far_range = op3::estimate_person_range(cv::Rect(100, 50, 40, 80), model);
  const double near_range = op3::estimate_person_range(cv::Rect(100, 50, 40, 180), model);

  EXPECT_GT(far_range, near_range);
}

TEST(EstimatePersonRangeTest, RoundTripsProjectedBoundingBox) {
  const op3::CameraModel model = op3::make_default_camera_model(op3::CameraPosition::kLeftFront);
  const double expected_range_m = 4.25;
  const int bbox_width =
      static_cast<int>(std::round(op3::project_person_width_pixels(expected_range_m, model)));
  const int bbox_height =
      static_cast<int>(std::round(op3::project_person_height_pixels(expected_range_m, model)));

  const double estimated_range =
      op3::estimate_person_range(cv::Rect(100, 50, bbox_width, bbox_height), model);

  EXPECT_NEAR(estimated_range, expected_range_m, 0.05);
}

TEST(EstimatePersonRangeTest, UsesWidthToBreakHeightTies) {
  const op3::CameraModel model = op3::make_default_camera_model(op3::CameraPosition::kLeftFront);

  const double narrower_range = op3::estimate_person_range(cv::Rect(100, 50, 42, 140), model);
  const double wider_range = op3::estimate_person_range(cv::Rect(100, 50, 74, 140), model);

  EXPECT_GT(narrower_range, wider_range);
}

TEST(NormalizeAngleTest, WrapsPastPositiveBoundary) {
  EXPECT_NEAR(op3::normalize_angle(225.0), -135.0, 1e-6);
}

TEST(MockPersonDetectorTest, ReturnsSingleDetectionForDarkBlob) {
  cv::Mat image(300, 400, CV_8UC3, cv::Scalar(235, 235, 235));
  const int hero_gray = op3::scene_label_grayscale("Hero");
  cv::rectangle(image, cv::Rect(120, 80, 50, 120), cv::Scalar(hero_gray, hero_gray, hero_gray),
                cv::FILLED);

  op3::MockPersonDetector detector;
  const std::vector<op3::Detection> detections = detector.detect(image);

  ASSERT_EQ(detections.size(), 1U);
  EXPECT_EQ(detections.front().id, "1");
  EXPECT_EQ(detections.front().label, "Hero");
  EXPECT_EQ(detections.front().bbox, cv::Rect(120, 80, 50, 120));
}

TEST(MockPersonDetectorTest, ReturnsMultipleDetectionsForSeparateBlobs) {
  cv::Mat image(300, 400, CV_8UC3, cv::Scalar(235, 235, 235));
  const int engineer_gray = op3::scene_label_grayscale("Engineer");
  const int sentry_gray = op3::scene_label_grayscale("sentry");
  cv::rectangle(image, cv::Rect(60, 90, 40, 100),
                cv::Scalar(engineer_gray, engineer_gray, engineer_gray), cv::FILLED);
  cv::rectangle(image, cv::Rect(240, 70, 50, 120),
                cv::Scalar(sentry_gray, sentry_gray, sentry_gray), cv::FILLED);

  op3::MockPersonDetector detector;
  const std::vector<op3::Detection> detections = detector.detect(image);

  ASSERT_EQ(detections.size(), 2U);
  EXPECT_EQ(detections[0].label, "Engineer");
  EXPECT_EQ(detections[0].bbox, cv::Rect(60, 90, 40, 100));
  EXPECT_EQ(detections[1].label, "sentry");
  EXPECT_EQ(detections[1].bbox, cv::Rect(240, 70, 50, 120));
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
              op3::Detection{.id = "cam", .bbox = cv::Rect(80, 20, 40, 100), .confidence = 0.9F},
          },
          &call_count),
      input_queue, detection_queue);
  worker.start();

  const auto timestamp = std::chrono::steady_clock::now();
  input_queue.push(op3::FrameMessage{
      .camera = op3::CameraPosition::kLeftFront,
      .frame_id = 11,
      .timestamp = timestamp,
      .robot_pose = {},
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
  EXPECT_DOUBLE_EQ(message.robot_pose.x_m, 0.0);
  ASSERT_EQ(message.person.size(), 1U);
  EXPECT_EQ(message.person.front().camera, op3::CameraPosition::kLeftFront);
  EXPECT_TRUE(message.person.front().label.empty());
  EXPECT_GT(message.person.front().range_m, 0.0);
  EXPECT_GT(message.person.front().x_m, 0.0);
  EXPECT_GT(message.person.front().world_x_m, 0.0);
}

TEST(CameraWorkerTest, OffsetsBodyPointByCameraMount) {
  op3::BlockingQueue<op3::FrameMessage> input_queue(1);
  op3::BlockingQueue<op3::DetectionMessage> detection_queue(4);

  op3::CameraModel model = op3::make_default_camera_model(op3::CameraPosition::kLeftFront);
  const int bbox_width = static_cast<int>(std::round(op3::project_person_width_pixels(4.0, model)));
  const int bbox_height =
      static_cast<int>(std::round(op3::project_person_height_pixels(4.0, model)));
  const int bbox_x = (model.image_width / 2) - (bbox_width / 2);
  op3::CameraWorker worker(
      model,
      std::make_unique<FixedDetector>(
          std::vector<op3::Detection>{
              op3::Detection{.id = "cam",
                             .bbox = cv::Rect(bbox_x, 40, bbox_width, bbox_height),
                             .confidence = 0.9F},
          },
          nullptr),
      input_queue, detection_queue);
  worker.start();

  input_queue.push(op3::FrameMessage{
      .camera = op3::CameraPosition::kLeftFront,
      .frame_id = 12,
      .timestamp = std::chrono::steady_clock::now(),
      .robot_pose = {.x_m = 10.0, .y_m = 20.0, .yaw_degrees = 0.0},
      .image = cv::Mat(model.image_height, model.image_width, CV_8UC3, cv::Scalar(235, 235, 235)),
  });
  input_queue.close();
  worker.join();

  detection_queue.close();
  op3::DetectionMessage message{};
  ASSERT_TRUE(detection_queue.pop(message));
  ASSERT_EQ(message.person.size(), 1U);

  const double camera_range_m =
      op3::estimate_person_range(cv::Rect(bbox_x, 40, bbox_width, bbox_height), model);
  const cv::Point2d ray = op3::body_frame_point_from_polar(camera_range_m, 45.0);
  const cv::Point2d expected_body{model.mount_x_m + ray.x, model.mount_y_m + ray.y};

  EXPECT_NEAR(message.person.front().x_m, expected_body.x, 0.05);
  EXPECT_NEAR(message.person.front().y_m, expected_body.y, 0.05);
  EXPECT_NEAR(message.person.front().world_x_m, 10.0 + expected_body.x, 0.05);
  EXPECT_NEAR(message.person.front().world_y_m, 20.0 + expected_body.y, 0.05);
  EXPECT_NEAR(message.person.front().range_m, std::hypot(expected_body.x, expected_body.y), 0.05);
}

TEST(SimulationPipelineTest, EstimatesRangeCloseToRenderedPersonPosition) {
  op3::SimulationConfig config;
  config.map_size = {20.0, 20.0};
  config.camera_mounts = {
      {op3::CameraPosition::kLeftFront, 45.0, {0.35, 0.22}},
  };
  config.pedestrian_waypoints = {
      {"Hero", {{13.18, 13.05}}, 0.0},
  };

  op3::SimulationEngine simulation(config);
  const op3::RobotState robot = simulation.robot_state();
  const std::vector<op3::PedestrianState> pedestrians = simulation.pedestrian_states();
  const std::vector<op3::FrameInput> frames = simulation.render_frames();

  ASSERT_EQ(frames.size(), 1U);
  ASSERT_EQ(pedestrians.size(), 1U);

  op3::BlockingQueue<op3::FrameMessage> input_queue(1);
  op3::BlockingQueue<op3::DetectionMessage> detection_queue(2);
  op3::CameraModel model = op3::make_default_camera_model(op3::CameraPosition::kLeftFront,
                                                          config.camera_image_size.width,
                                                          config.camera_image_size.height);
  op3::CameraWorker worker(model, std::make_unique<op3::MockPersonDetector>(), input_queue,
                           detection_queue);
  worker.start();

  input_queue.push(op3::FrameMessage{
      .camera = op3::CameraPosition::kLeftFront,
      .frame_id = 1,
      .timestamp = std::chrono::steady_clock::now(),
      .robot_pose = {.x_m = robot.position.x,
                     .y_m = robot.position.y,
                     .yaw_degrees = robot.yaw_degrees},
      .image = frames.front().image,
  });
  input_queue.close();
  worker.join();

  detection_queue.close();
  op3::DetectionMessage message{};
  ASSERT_TRUE(detection_queue.pop(message));
  ASSERT_EQ(message.person.size(), 1U);
  EXPECT_EQ(message.person.front().label, "Hero");
  EXPECT_NEAR(message.person.front().world_x_m, pedestrians.front().position.x, 0.35);
  EXPECT_NEAR(message.person.front().world_y_m, pedestrians.front().position.y, 0.35);
  EXPECT_NEAR(message.person.front().range_m,
              std::hypot(pedestrians.front().position.x - robot.position.x,
                         pedestrians.front().position.y - robot.position.y),
              0.35);
}

TEST(FusionTrackerTest, AssociatesAsynchronousDetectionsIntoSingleTrack) {
  op3::BlockingQueue<op3::DetectionMessage> detection_queue(8);
  op3::FusionTracker tracker(detection_queue, 1.0, 0.45, 0.3);
  tracker.start();

  const auto base_time = std::chrono::steady_clock::now();
  detection_queue.push(op3::DetectionMessage{
      .camera = op3::CameraPosition::kLeftFront,
      .frame_id = 1,
      .timestamp = base_time,
      .robot_pose = {},
      .person = {make_report("1", 30.0, 5.0, op3::CameraPosition::kLeftFront)},
  });
  detection_queue.push(op3::DetectionMessage{
      .camera = op3::CameraPosition::kRightFront,
      .frame_id = 2,
      .timestamp = base_time + std::chrono::milliseconds(15),
      .robot_pose = {},
      .person = {make_report("1", 34.0, 5.2, op3::CameraPosition::kRightFront)},
  });

  allow_tracker_to_drain();
  tracker.stop();
  tracker.join();

  const op3::PipelineOutput output = tracker.snapshot();
  EXPECT_EQ(output.sequence_id, 2U);
  ASSERT_EQ(output.person.size(), 1U);
  EXPECT_EQ(output.person.front().track_id, "track-1");
  EXPECT_TRUE(output.person.front().label.empty());
  EXPECT_GT(output.person.front().x_m, 4.0);
  EXPECT_GT(output.person.front().world_x_m, 4.0);
  EXPECT_GT(output.person.front().range_m, 4.5);
  EXPECT_LT(output.person.front().radius_m, 1.0);
  std::vector<op3::CameraPosition> sources = output.person.front().sources;
  std::sort(sources.begin(), sources.end(), [](op3::CameraPosition lhs, op3::CameraPosition rhs) {
    return static_cast<int>(lhs) < static_cast<int>(rhs);
  });
  ASSERT_EQ(sources.size(), 2U);
  EXPECT_EQ(sources[0], op3::CameraPosition::kLeftFront);
  EXPECT_EQ(sources[1], op3::CameraPosition::kRightFront);
}

TEST(FusionTrackerTest, StabilizesAngularVelocityAcrossConsistentMeasurements) {
  op3::BlockingQueue<op3::DetectionMessage> detection_queue(8);
  op3::FusionTracker tracker(detection_queue, 1.5, 0.45, 0.3);
  tracker.start();

  const auto base_time = std::chrono::steady_clock::now();
  detection_queue.push(op3::DetectionMessage{
      .camera = op3::CameraPosition::kLeftFront,
      .frame_id = 1,
      .timestamp = base_time,
      .robot_pose = {},
      .person = {make_report("1", 0.0, 4.0, op3::CameraPosition::kLeftFront)},
  });
  detection_queue.push(op3::DetectionMessage{
      .camera = op3::CameraPosition::kLeftFront,
      .frame_id = 2,
      .timestamp = base_time + std::chrono::milliseconds(50),
      .robot_pose = {},
      .person = {make_report("1", 0.0, 4.6, op3::CameraPosition::kLeftFront)},
  });
  detection_queue.push(op3::DetectionMessage{
      .camera = op3::CameraPosition::kLeftFront,
      .frame_id = 3,
      .timestamp = base_time + std::chrono::milliseconds(100),
      .robot_pose = {},
      .person = {make_report("1", 0.0, 5.2, op3::CameraPosition::kLeftFront)},
  });

  allow_tracker_to_drain();
  tracker.stop();
  tracker.join();

  const op3::PipelineOutput output = tracker.snapshot();
  ASSERT_EQ(output.person.size(), 1U);
  EXPECT_GT(output.person.front().x_m, 4.3);
  EXPECT_GT(output.person.front().vx_mps, 0.5);
  EXPECT_GT(output.person.front().confidence, 0.75);
}

TEST(FusionTrackerTest, PrefersStableLabelIdentityOverDistanceGate) {
  op3::BlockingQueue<op3::DetectionMessage> detection_queue(8);
  op3::FusionTracker tracker(detection_queue, 0.5, 0.45, 0.3);
  tracker.start();

  const auto base_time = std::chrono::steady_clock::now();
  detection_queue.push(op3::DetectionMessage{
      .camera = op3::CameraPosition::kLeftFront,
      .frame_id = 1,
      .timestamp = base_time,
      .robot_pose = {},
      .person = {make_report("hero-1", 20.0, 4.0, op3::CameraPosition::kLeftFront, "Hero")},
  });
  detection_queue.push(op3::DetectionMessage{
      .camera = op3::CameraPosition::kRightFront,
      .frame_id = 2,
      .timestamp = base_time + std::chrono::milliseconds(20),
      .robot_pose = {},
      .person = {make_report("hero-1", 32.0, 5.3, op3::CameraPosition::kRightFront, "Hero")},
  });

  allow_tracker_to_drain();
  tracker.stop();
  tracker.join();

  const op3::PipelineOutput output = tracker.snapshot();
  ASSERT_EQ(output.person.size(), 1U);
  EXPECT_EQ(output.person.front().label, "Hero");
}

TEST(FusionTrackerTest, SeparatesTracksOutsideAssociationGate) {
  op3::BlockingQueue<op3::DetectionMessage> detection_queue(8);
  op3::FusionTracker tracker(detection_queue, 1.0, 0.45, 0.3);
  tracker.start();

  detection_queue.push(op3::DetectionMessage{
      .camera = op3::CameraPosition::kLeftFront,
      .frame_id = 1,
      .timestamp = std::chrono::steady_clock::now(),
      .robot_pose = {},
      .person =
          {
              make_report("1", 15.0, 4.0, op3::CameraPosition::kLeftFront),
              make_report("2", 60.0, 4.0, op3::CameraPosition::kLeftFront),
          },
  });

  allow_tracker_to_drain();
  tracker.stop();
  tracker.join();

  const op3::PipelineOutput output = tracker.snapshot();
  ASSERT_EQ(output.person.size(), 2U);
}

TEST(FusionTrackerTest, DoesNotAssignTwoReportsToOneTrackInSingleMessage) {
  op3::BlockingQueue<op3::DetectionMessage> detection_queue(8);
  op3::FusionTracker tracker(detection_queue, 0.6, 0.45, 0.3);
  tracker.start();

  const auto base_time = std::chrono::steady_clock::now();
  detection_queue.push(op3::DetectionMessage{
      .camera = op3::CameraPosition::kLeftFront,
      .frame_id = 1,
      .timestamp = base_time,
      .robot_pose = {},
      .person = {make_report("1", 30.0, 5.0, op3::CameraPosition::kLeftFront)},
  });
  detection_queue.push(op3::DetectionMessage{
      .camera = op3::CameraPosition::kLeftFront,
      .frame_id = 2,
      .timestamp = base_time + std::chrono::milliseconds(10),
      .robot_pose = {},
      .person =
          {
              make_report("1", 31.0, 5.0, op3::CameraPosition::kLeftFront),
              make_report("2", 32.0, 5.0, op3::CameraPosition::kLeftFront),
          },
  });

  allow_tracker_to_drain();
  tracker.stop();
  tracker.join();

  const op3::PipelineOutput output = tracker.snapshot();
  ASSERT_EQ(output.person.size(), 2U);
}

TEST(FusionTrackerTest, KeepsOnlyOneTrackPerLabel) {
  op3::BlockingQueue<op3::DetectionMessage> detection_queue(8);
  op3::FusionTracker tracker(detection_queue, 0.3, 0.45, 0.3);
  tracker.start();

  const auto base_time = std::chrono::steady_clock::now();
  detection_queue.push(op3::DetectionMessage{
      .camera = op3::CameraPosition::kLeftFront,
      .frame_id = 1,
      .timestamp = base_time,
      .robot_pose = {},
      .person =
          {
              make_report("hero-a", 20.0, 4.0, op3::CameraPosition::kLeftFront, "Hero"),
              make_report("hero-b", 38.0, 5.5, op3::CameraPosition::kLeftFront, "Hero"),
          },
  });

  allow_tracker_to_drain();
  tracker.stop();
  tracker.join();

  const op3::PipelineOutput output = tracker.snapshot();
  ASSERT_EQ(output.person.size(), 1U);
  EXPECT_EQ(output.person.front().label, "Hero");
}

TEST(FusionTrackerTest, RemovesStaleTracksAfterTimeout) {
  op3::BlockingQueue<op3::DetectionMessage> detection_queue(8);
  op3::FusionTracker tracker(detection_queue, 1.0, 0.45, 0.3);
  tracker.start();

  const auto base_time = std::chrono::steady_clock::now();
  detection_queue.push(op3::DetectionMessage{
      .camera = op3::CameraPosition::kLeftFront,
      .frame_id = 1,
      .timestamp = base_time,
      .robot_pose = {},
      .person = {make_report("1", 40.0, 5.0, op3::CameraPosition::kLeftFront)},
  });
  detection_queue.push(op3::DetectionMessage{
      .camera = op3::CameraPosition::kRightFront,
      .frame_id = 2,
      .timestamp = base_time + std::chrono::milliseconds(1500),
      .robot_pose = {},
      .person = {},
  });

  allow_tracker_to_drain();
  tracker.stop();
  tracker.join();

  const op3::PipelineOutput output = tracker.snapshot();
  EXPECT_TRUE(output.person.empty());
}

TEST(FusionTrackerTest, HandlesAssociationAcrossAngleWrapBoundary) {
  op3::BlockingQueue<op3::DetectionMessage> detection_queue(8);
  op3::FusionTracker tracker(detection_queue, 0.5, 0.45, 0.3);
  tracker.start();

  const auto base_time = std::chrono::steady_clock::now();
  detection_queue.push(op3::DetectionMessage{
      .camera = op3::CameraPosition::kLeftRear,
      .frame_id = 1,
      .timestamp = base_time,
      .robot_pose = {},
      .person = {make_report("1", 179.0, 5.0, op3::CameraPosition::kLeftRear)},
  });
  detection_queue.push(op3::DetectionMessage{
      .camera = op3::CameraPosition::kRightRear,
      .frame_id = 2,
      .timestamp = base_time + std::chrono::milliseconds(20),
      .robot_pose = {},
      .person = {make_report("1", -179.0, 5.0, op3::CameraPosition::kRightRear)},
  });

  allow_tracker_to_drain();
  tracker.stop();
  tracker.join();

  const op3::PipelineOutput output = tracker.snapshot();
  ASSERT_EQ(output.person.size(), 1U);
  EXPECT_LT(std::abs(op3::normalize_angle(output.person.front().angle - (-179.0))), 3.0);
  EXPECT_LT(std::abs(output.person.front().y_m), 0.3);
}

TEST(FusionTrackerTest, KeepsWorldPositionStableWhenRobotRotates) {
  op3::BlockingQueue<op3::DetectionMessage> detection_queue(8);
  op3::FusionTracker tracker(detection_queue, 1.0, 0.45, 0.3);
  tracker.start();

  const auto base_time = std::chrono::steady_clock::now();
  detection_queue.push(op3::DetectionMessage{
      .camera = op3::CameraPosition::kLeftFront,
      .frame_id = 1,
      .timestamp = base_time,
      .robot_pose = {.x_m = 10.0, .y_m = 10.0, .yaw_degrees = 0.0},
      .person = {op3::PersonReport{
          .id = "1",
          .angle = 0.0,
          .range_m = 5.0,
          .x_m = 5.0,
          .y_m = 0.0,
          .world_x_m = 15.0,
          .world_y_m = 10.0,
          .camera = op3::CameraPosition::kLeftFront,
      }},
  });
  detection_queue.push(op3::DetectionMessage{
      .camera = op3::CameraPosition::kLeftFront,
      .frame_id = 2,
      .timestamp = base_time + std::chrono::milliseconds(30),
      .robot_pose = {.x_m = 10.0, .y_m = 10.0, .yaw_degrees = 90.0},
      .person = {op3::PersonReport{
          .id = "1",
          .angle = -90.0,
          .range_m = 5.0,
          .x_m = 0.0,
          .y_m = -5.0,
          .world_x_m = 15.0,
          .world_y_m = 10.0,
          .camera = op3::CameraPosition::kLeftFront,
      }},
  });

  allow_tracker_to_drain();
  tracker.stop();
  tracker.join();

  const op3::PipelineOutput output = tracker.snapshot();
  ASSERT_EQ(output.person.size(), 1U);
  EXPECT_NEAR(output.person.front().world_x_m, 15.0, 0.25);
  EXPECT_NEAR(output.person.front().world_y_m, 10.0, 0.25);
  EXPECT_NEAR(output.person.front().x_m, 0.0, 0.35);
  EXPECT_NEAR(output.person.front().y_m, -5.0, 0.35);
}

TEST(StatePublisherTest, PublishesTrackerSnapshots) {
  op3::BlockingQueue<op3::DetectionMessage> detection_queue(8);
  op3::FusionTracker tracker(detection_queue);
  tracker.start();

  std::atomic<int> publish_count = 0;
  op3::StatePublisher publisher(tracker, std::chrono::milliseconds(10),
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
      .robot_pose = {},
      .person = {make_report("1", 30.0, 5.0, op3::CameraPosition::kLeftFront)},
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
          .label = "Hero",
          .x_m = 4.25,
          .y_m = 2.45,
          .world_x_m = 14.25,
          .world_y_m = 12.45,
          .vx_mps = 0.7,
          .vy_mps = -0.1,
          .range_m = 4.91,
          .radius_m = 0.65,
          .angle = 30.0,
          .angle_velocity = 1.5,
          .confidence = 0.95,
          .sources = {op3::CameraPosition::kLeftFront},
          .last_update = timestamp,
      }},
  };

  const std::string json = op3::to_json(output);

  EXPECT_EQ(json,
            "{\"sequence_id\":9,\"timestamp_ms\":300,\"person\":[{\"track_id\":\"track-1\","
            "\"label\":\"Hero\",\"x_m\":4.25,\"y_m\":2.45,\"world_x_m\":14.25,\"world_y_m\":12.45,"
            "\"vx_mps\":0.70,\"vy_mps\":-0.10,\"range_m\":4.91,\"radius_m\":0.65,"
            "\"angle\":30.0,\"angle_velocity\":1.5,\"confidence\":0.95,"
            "\"sources\":[\"left_front\"],\"last_update_ms\":250}]}");
}

} // namespace
