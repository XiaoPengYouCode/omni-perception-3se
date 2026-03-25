#include <string>

#include "logger.hpp"
#include <opencv2/opencv.hpp>

namespace {

constexpr int kImageWidth = 960;
constexpr int kImageHeight = 540;
constexpr int kSquareSize = 60;

cv::Mat make_demo_image() {
  cv::Mat image(kImageHeight, kImageWidth, CV_8UC3, cv::Scalar(235, 235, 235));

  for (int y = 0; y < kImageHeight; y += kSquareSize) {
    for (int x = 0; x < kImageWidth; x += kSquareSize) {
      const bool is_dark_square = ((x / kSquareSize) + (y / kSquareSize)) % 2 == 0;
      const cv::Scalar color = is_dark_square ? cv::Scalar(40, 40, 40)
                                              : cv::Scalar(245, 245, 245);
      cv::rectangle(image, cv::Rect(x, y, kSquareSize, kSquareSize), color, cv::FILLED);
    }
  }

  return image;
}

}  // namespace

int main() {
  cmake_demo::log_info("Creating demo image: {}x{}", kImageWidth, kImageHeight);
  cv::Mat image = make_demo_image();

  if (image.empty()) {
    cmake_demo::log_error("Failed to create demo image");
    return 1;
  }

  const std::string window_name = "cmake-demo";
  cmake_demo::log_info("Opening window '{}'", window_name);
  cmake_demo::log_info("Press q or Esc to quit");

  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
  cv::imshow(window_name, image);

  while (true) {
    const int key = cv::waitKey(16);
    if (key == 27 || key == 'q' || key == 'Q') {
      cmake_demo::log_info("Exit requested by user");
      break;
    }
  }

  cv::destroyAllWindows();
  cmake_demo::log_info("Demo finished cleanly");
  return 0;
}
