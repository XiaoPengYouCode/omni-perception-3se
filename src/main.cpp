#include "core/blocking_queue.hpp"
#include "perception/angle_utils.hpp"
#include "perception/camera_worker.hpp"
#include "perception/detector_factory.hpp"
#include "perception/fusion_tracker.hpp"
#include "perception/json_output.hpp"
#include "perception/state_publisher.hpp"
#include "perception/types.hpp"
#include "simulation/simulation_engine.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <vector>

#include <fmt/format.h>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace {

constexpr char kControlWindowName[] = "simulation-controls";
constexpr double kPi = 3.14159265358979323846;

op3::SimulationControls controls_from_key(int key) {
  op3::SimulationControls controls;
  switch (key) {
  case 'w':
  case 'W':
    controls.move_forward = true;
    break;
  case 's':
  case 'S':
    controls.move_backward = true;
    break;
  case 'a':
  case 'A':
    controls.move_left = true;
    break;
  case 'd':
  case 'D':
    controls.move_right = true;
    break;
  case 'q':
  case 'Q':
    controls.rotate_left = true;
    break;
  case 'e':
  case 'E':
    controls.rotate_right = true;
    break;
  default:
    break;
  }
  return controls;
}

cv::Mat make_control_overlay(const op3::RobotState& robot, double sim_time_sec) {
  cv::Mat image(150, 560, CV_8UC3, cv::Scalar(28, 28, 28));
  const int font = cv::FONT_HERSHEY_SIMPLEX;
  const cv::Scalar text_color(230, 230, 230);
  cv::putText(image, "W/S forward  A/D strafe  Q/E rotate  ESC quit", {16, 40}, font, 0.7,
              text_color, 2, cv::LINE_AA);
  cv::putText(image,
              fmt::format("robot x={:.2f} y={:.2f} yaw={:.1f}", robot.position.x, robot.position.y,
                          robot.yaw_degrees),
              {16, 86}, font, 0.65, cv::Scalar(170, 220, 255), 2, cv::LINE_AA);
  cv::putText(image, fmt::format("sim t={:.2f}s", sim_time_sec), {16, 124}, font, 0.65,
              cv::Scalar(160, 240, 160), 2, cv::LINE_AA);
  return image;
}

cv::Point world_to_canvas(const cv::Point2d& position, const cv::Size& canvas_size,
                          const cv::Size2d& map_size) {
  const double x = std::clamp(position.x / std::max(map_size.width, 1.0), 0.0, 1.0);
  const double y = std::clamp(position.y / std::max(map_size.height, 1.0), 0.0, 1.0);
  const int canvas_x = static_cast<int>(x * static_cast<double>(canvas_size.width - 1));
  const int canvas_y = static_cast<int>((1.0 - y) * static_cast<double>(canvas_size.height - 1));
  return {canvas_x, canvas_y};
}

double meters_to_pixels(double meters, int canvas_extent, double world_extent) {
  if (world_extent <= 0.0) {
    return 0.0;
  }
  return meters * (static_cast<double>(canvas_extent) / world_extent);
}

cv::Point2d rotate_world_offset(const cv::Point2d& offset, double yaw_degrees) {
  const double yaw_rad = yaw_degrees * (kPi / 180.0);
  const double cos_yaw = std::cos(yaw_rad);
  const double sin_yaw = std::sin(yaw_rad);
  return {
      (offset.x * cos_yaw) - (offset.y * sin_yaw),
      (offset.x * sin_yaw) + (offset.y * cos_yaw),
  };
}

cv::Mat make_topdown_panel(const op3::SimulationConfig& config, const op3::RobotState& robot,
                           const std::vector<op3::PedestrianState>& pedestrians,
                           const op3::PipelineOutput& fused_output) {
  cv::Mat image(620, 620, CV_8UC3, cv::Scalar(245, 247, 250));
  const cv::Scalar grid_color(224, 228, 235);
  const cv::Scalar border_color(180, 188, 200);
  const cv::Scalar robot_color(72, 142, 255);
  const cv::Scalar gt_color(48, 176, 96);
  const cv::Scalar track_color(255, 142, 48);
  const cv::Scalar text_color(48, 56, 72);
  const int margin = 28;
  const cv::Rect map_rect(margin, margin + 28, image.cols - (margin * 2), image.rows - 96);

  cv::rectangle(image, map_rect, cv::Scalar(255, 255, 255), cv::FILLED);
  cv::rectangle(image, map_rect, border_color, 2);
  for (int index = 1; index < 10; ++index) {
    const int x = map_rect.x + ((map_rect.width * index) / 10);
    const int y = map_rect.y + ((map_rect.height * index) / 10);
    cv::line(image, {x, map_rect.y}, {x, map_rect.br().y}, grid_color, 1);
    cv::line(image, {map_rect.x, y}, {map_rect.br().x, y}, grid_color, 1);
  }

  cv::putText(image, "Topdown Scene", {margin, margin}, cv::FONT_HERSHEY_SIMPLEX, 0.8, text_color,
              2, cv::LINE_AA);

  const auto to_map_canvas = [&](const cv::Point2d& position) {
    const cv::Point local = world_to_canvas(position, map_rect.size(), config.map_size);
    return cv::Point(map_rect.x + local.x, map_rect.y + local.y);
  };

  for (const op3::PedestrianState& pedestrian : pedestrians) {
    const cv::Point center = to_map_canvas(pedestrian.position);
    cv::circle(image, center, 7, gt_color, cv::FILLED, cv::LINE_AA);
    cv::putText(image, pedestrian.id, center + cv::Point(10, -10), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                gt_color, 1, cv::LINE_AA);
  }

  for (const op3::TrackedPerson& person : fused_output.person) {
    const cv::Point center = to_map_canvas({person.world_x_m, person.world_y_m});
    const int radius_px =
        std::max(8, static_cast<int>(std::round(
                        meters_to_pixels(person.radius_m, map_rect.width, config.map_size.width))));
    cv::circle(image, center, radius_px, track_color, 2, cv::LINE_AA);
    cv::circle(image, center, 4, track_color, cv::FILLED, cv::LINE_AA);
    const std::string display_name = person.label.empty() ? person.track_id : person.label;
    cv::putText(image, display_name, center + cv::Point(10, 16), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                track_color, 1, cv::LINE_AA);
  }

  const cv::Point robot_center = to_map_canvas(robot.position);
  cv::circle(image, robot_center, 10, robot_color, cv::FILLED, cv::LINE_AA);
  const double yaw_rad = robot.yaw_degrees * (kPi / 180.0);
  const cv::Point heading_end(robot_center.x + static_cast<int>(std::cos(yaw_rad) * 30.0),
                              robot_center.y - static_cast<int>(std::sin(yaw_rad) * 30.0));
  cv::arrowedLine(image, robot_center, heading_end, robot_color, 2, cv::LINE_AA, 0, 0.25);

  for (const op3::SimulationConfig::CameraMount& mount : config.camera_mounts) {
    const double mount_yaw_rad = (robot.yaw_degrees + mount.yaw_offset_degrees) * (kPi / 180.0);
    const cv::Point2d mount_position =
        robot.position + rotate_world_offset(mount.mount_offset, robot.yaw_degrees);
    const cv::Point mount_center = to_map_canvas(mount_position);
    const cv::Point mount_end(mount_center.x + static_cast<int>(std::cos(mount_yaw_rad) * 22.0),
                              mount_center.y - static_cast<int>(std::sin(mount_yaw_rad) * 22.0));
    cv::arrowedLine(image, mount_center, mount_end, cv::Scalar(145, 150, 180), 1, cv::LINE_AA, 0,
                    0.2);
  }

  cv::putText(image,
              fmt::format("GT pedestrians: {}   fused tracks: {}", pedestrians.size(),
                          fused_output.person.size()),
              {margin, image.rows - 24}, cv::FONT_HERSHEY_SIMPLEX, 0.58, text_color, 2,
              cv::LINE_AA);
  return image;
}

cv::Mat make_camera_grid(const std::vector<op3::FrameInput>& frames) {
  constexpr int tile_width = 320;
  constexpr int tile_height = 180;
  cv::Mat canvas((tile_height * 2) + 8, (tile_width * 2) + 8, CV_8UC3, cv::Scalar(30, 30, 30));

  const std::array<op3::CameraPosition, 4> order = {
      op3::CameraPosition::kLeftFront,
      op3::CameraPosition::kRightFront,
      op3::CameraPosition::kLeftRear,
      op3::CameraPosition::kRightRear,
  };

  for (std::size_t index = 0; index < order.size(); ++index) {
    const auto it = std::find_if(
        frames.begin(), frames.end(),
        [camera = order[index]](const op3::FrameInput& frame) { return frame.camera == camera; });
    if (it == frames.end() || it->image.empty()) {
      continue;
    }

    cv::Mat tile;
    cv::resize(it->image, tile, cv::Size(tile_width, tile_height));
    const int tile_x = static_cast<int>(index % 2) * (tile_width + 8);
    const int tile_y = static_cast<int>(index / 2) * (tile_height + 8);
    tile.copyTo(canvas(cv::Rect(tile_x, tile_y, tile_width, tile_height)));
    cv::putText(canvas, op3::camera_position_to_string(order[index]), {tile_x + 10, tile_y + 24},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  }

  return canvas;
}

cv::Mat make_display_panel(const op3::SimulationConfig& config, const op3::RobotState& robot,
                           const std::vector<op3::PedestrianState>& pedestrians,
                           const std::vector<op3::FrameInput>& frames,
                           const op3::PipelineOutput& fused_output, double sim_time_sec) {
  const cv::Mat topdown = make_topdown_panel(config, robot, pedestrians, fused_output);
  const cv::Mat camera_grid = make_camera_grid(frames);
  const cv::Mat controls = make_control_overlay(robot, sim_time_sec);

  const int width = std::max({topdown.cols, camera_grid.cols, controls.cols});
  const int height = topdown.rows + camera_grid.rows + controls.rows + 16;
  cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(24, 24, 24));

  topdown.copyTo(canvas(cv::Rect(0, 0, topdown.cols, topdown.rows)));
  camera_grid.copyTo(canvas(cv::Rect(0, topdown.rows + 8, camera_grid.cols, camera_grid.rows)));
  controls.copyTo(
      canvas(cv::Rect(0, topdown.rows + camera_grid.rows + 16, controls.cols, controls.rows)));
  return canvas;
}

} // namespace

int main() {
  setenv("OPENCV_LOG_LEVEL", "ERROR", 1);
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

  op3::SimulationConfig simulation_config = op3::SimulationEngine::default_config();
  op3::SimulationEngine simulation(simulation_config);

  op3::BlockingQueue<op3::FrameMessage> left_front_queue(2);
  op3::BlockingQueue<op3::FrameMessage> right_front_queue(2);
  op3::BlockingQueue<op3::FrameMessage> left_rear_queue(2);
  op3::BlockingQueue<op3::FrameMessage> right_rear_queue(2);
  op3::BlockingQueue<op3::DetectionMessage> detection_queue(64);

  op3::FusionTracker tracker(detection_queue);
  tracker.start();

  op3::StatePublisher publisher(tracker, std::chrono::milliseconds(50),
                                [](const op3::PipelineOutput& output) {
                                  if (output.sequence_id == 0) {
                                    return;
                                  }
                                  fmt::print("{}\n", op3::to_json(output));
                                });
  publisher.start();

  const auto make_camera_model = [&simulation_config](op3::CameraPosition camera) {
    op3::CameraModel model =
        op3::make_default_camera_model(camera, simulation_config.camera_image_size.width,
                                       simulation_config.camera_image_size.height);
    model.horizontal_fov_degrees = simulation_config.camera_horizontal_fov_degrees;
    model.assumed_person_height_m = simulation_config.assumed_person_height_m;
    model.max_range_m = simulation_config.max_person_range_m;
    return model;
  };

  op3::CameraWorker left_front_worker(
      make_camera_model(op3::CameraPosition::kLeftFront),
      op3::DetectorFactory::create_yolo26_nano(make_camera_model(op3::CameraPosition::kLeftFront)),
      left_front_queue, detection_queue);
  op3::CameraWorker right_front_worker(
      make_camera_model(op3::CameraPosition::kRightFront),
      op3::DetectorFactory::create_yolo26_nano(make_camera_model(op3::CameraPosition::kRightFront)),
      right_front_queue, detection_queue);
  op3::CameraWorker left_rear_worker(
      make_camera_model(op3::CameraPosition::kLeftRear),
      op3::DetectorFactory::create_yolo26_nano(make_camera_model(op3::CameraPosition::kLeftRear)),
      left_rear_queue, detection_queue);
  op3::CameraWorker right_rear_worker(
      make_camera_model(op3::CameraPosition::kRightRear),
      op3::DetectorFactory::create_yolo26_nano(make_camera_model(op3::CameraPosition::kRightRear)),
      right_rear_queue, detection_queue);

  left_front_worker.start();
  right_front_worker.start();
  left_rear_worker.start();
  right_rear_worker.start();

  cv::namedWindow(kControlWindowName, cv::WINDOW_AUTOSIZE);

  const auto start_time = std::chrono::steady_clock::now();
  auto next_tick = start_time;
  std::uint64_t frame_id = 1;
  double sim_time_sec = 0.0;
  bool running = true;

  while (running) {
    const int key = cv::waitKey(1);
    if (key == 27) {
      running = false;
      break;
    }

    simulation.update(simulation_config.tick_seconds, controls_from_key(key));
    const std::vector<op3::FrameInput> frames = simulation.render_frames();
    const op3::RobotState robot = simulation.robot_state();
    const std::vector<op3::PedestrianState> pedestrians = simulation.pedestrian_states();

    const auto timestamp =
        start_time + std::chrono::milliseconds(static_cast<int64_t>(sim_time_sec * 1000.0));
    for (const op3::FrameInput& frame : frames) {
      const op3::FrameMessage message{
          .camera = frame.camera,
          .frame_id = frame_id,
          .timestamp = timestamp,
          .robot_pose = op3::RobotPose{.x_m = robot.position.x,
                                       .y_m = robot.position.y,
                                       .yaw_degrees = robot.yaw_degrees},
          .image = frame.image,
      };

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
    }

    const op3::PipelineOutput fused_output = tracker.snapshot();
    cv::imshow(kControlWindowName, make_display_panel(simulation_config, robot, pedestrians, frames,
                                                      fused_output, sim_time_sec));

    ++frame_id;
    sim_time_sec += simulation_config.tick_seconds;
    next_tick +=
        std::chrono::milliseconds(static_cast<int>(simulation_config.tick_seconds * 1000.0));
    std::this_thread::sleep_until(next_tick);
  }

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

  cv::destroyAllWindows();
  return 0;
}
