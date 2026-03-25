#include <sstream>
#include <string>
#include <vector>

#include "cmake_demo/app_config.hpp"
#include "cmake_demo/logger.hpp"
#include <opencv2/opencv.hpp>

namespace {

constexpr int kTileWidth = 640;
constexpr int kTileHeight = 360;

cv::Mat make_placeholder(int camera_id, const std::string& message) {
  cv::Mat frame(kTileHeight, kTileWidth, CV_8UC3, cv::Scalar(32, 32, 32));
  std::ostringstream title;
  title << "Camera " << camera_id;

  cv::putText(frame, title.str(), {24, 48}, cv::FONT_HERSHEY_SIMPLEX, 1.0,
              cv::Scalar(0, 200, 255), 2, cv::LINE_AA);
  cv::putText(frame, message, {24, 96}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
              cv::Scalar(220, 220, 220), 2, cv::LINE_AA);
  return frame;
}

cv::Mat fit_frame(const cv::Mat& source) {
  if (source.empty()) {
    return cv::Mat(kTileHeight, kTileWidth, CV_8UC3, cv::Scalar(0, 0, 0));
  }

  cv::Mat resized;
  cv::resize(source, resized, cv::Size(kTileWidth, kTileHeight));
  return resized;
}

cv::Mat compose_grid(const std::vector<cv::Mat>& tiles) {
  const int tile_count = static_cast<int>(tiles.size());
  const cmake_demo::GridShape grid = cmake_demo::compute_grid_shape(tile_count);

  cv::Mat canvas(grid.rows * kTileHeight, grid.columns * kTileWidth, CV_8UC3,
                 cv::Scalar(18, 18, 18));

  for (int i = 0; i < tile_count; ++i) {
    const int row = i / grid.columns;
    const int column = i % grid.columns;
    const cv::Rect roi(column * kTileWidth, row * kTileHeight, kTileWidth, kTileHeight);
    fit_frame(tiles[i]).copyTo(canvas(roi));
  }

  return canvas;
}

}  // namespace

int main(int argc, char* argv[]) {
  std::vector<std::string> args;
  args.reserve(argc > 1 ? argc - 1 : 0);
  for (int i = 1; i < argc; ++i) {
    args.emplace_back(argv[i]);
  }

  std::vector<int> camera_ids;
  try {
    camera_ids = cmake_demo::parse_camera_ids(args);
  } catch (const std::exception& ex) {
    cmake_demo::log_error("Invalid camera arguments: {}", ex.what());
    return 1;
  }

  std::vector<cv::VideoCapture> captures;
  captures.reserve(camera_ids.size());

  for (int camera_id : camera_ids) {
    cv::VideoCapture capture(camera_id);
    if (capture.isOpened()) {
      capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
      capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
      cmake_demo::log_info("Opened camera {}", camera_id);
    } else {
      cmake_demo::log_warn("Failed to open camera {}", camera_id);
    }
    captures.push_back(std::move(capture));
  }

  if (captures.empty()) {
    cmake_demo::log_error("No camera configured");
    return 1;
  }

  cmake_demo::log_info("Starting viewer with {} camera slot(s)", captures.size());
  cv::namedWindow("Multi Camera Viewer", cv::WINDOW_NORMAL);

  while (true) {
    std::vector<cv::Mat> tiles;
    tiles.reserve(captures.size());

    for (std::size_t i = 0; i < captures.size(); ++i) {
      cv::Mat frame;
      if (captures[i].isOpened() && captures[i].read(frame) && !frame.empty()) {
        cv::putText(frame, "Camera " + std::to_string(camera_ids[i]), {20, 40},
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2,
                    cv::LINE_AA);
        tiles.push_back(frame);
      } else {
        tiles.push_back(make_placeholder(camera_ids[i], "No signal"));
      }
    }

    cv::imshow("Multi Camera Viewer", compose_grid(tiles));

    const int key = cv::waitKey(1);
    if (key == 27 || key == 'q' || key == 'Q') {
      cmake_demo::log_info("Shutdown requested by user");
      break;
    }
  }

  return 0;
}
