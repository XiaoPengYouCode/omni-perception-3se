#include "perception/demo_frames.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace op3 {

namespace {

constexpr int kPersonRectWidth = 80;
constexpr int kPersonRectHeight = 220;
const cv::Scalar kBackgroundColor(235, 235, 235);
const cv::Scalar kForegroundColor(30, 30, 30);

/**
 * Draws a simple dark rectangle that stands in for a person in the synthetic demo frames.
 */
void draw_mock_person(cv::Mat& image, int center_x) {
  const int clamped_center_x =
      std::clamp(center_x, kPersonRectWidth / 2, image.cols - (kPersonRectWidth / 2));
  const int left = clamped_center_x - (kPersonRectWidth / 2);
  const int top = std::max(0, image.rows / 2 - kPersonRectHeight / 2);
  const int height = std::min(kPersonRectHeight, image.rows - top);

  cv::rectangle(image, cv::Rect(left, top, kPersonRectWidth, height), kForegroundColor, cv::FILLED);
}

} // namespace

std::vector<FrameInput> make_demo_frames(int image_width, int image_height) {
  std::vector<FrameInput> frames;
  frames.reserve(4);

  // Position fake people differently per camera so the downstream fusion output is non-trivial.
  const std::vector<std::pair<CameraPosition, int>> camera_layout = {
      {CameraPosition::kLeftFront, image_width / 3},
      {CameraPosition::kRightFront, (image_width * 2) / 3},
      {CameraPosition::kLeftRear, image_width / 2},
      {CameraPosition::kRightRear, -1},
  };

  for (const auto& [camera, person_center_x] : camera_layout) {
    cv::Mat image(image_height, image_width, CV_8UC3, kBackgroundColor);
    if (person_center_x >= 0) {
      draw_mock_person(image, person_center_x);
    }
    frames.push_back(FrameInput{.camera = camera, .image = image});
  }

  return frames;
}

} // namespace op3
