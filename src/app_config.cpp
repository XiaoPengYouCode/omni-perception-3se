#include "cmake_demo/app_config.hpp"

#include <cmath>
#include <stdexcept>

namespace cmake_demo {

std::vector<int> parse_camera_ids(const std::vector<std::string>& args,
                                  int default_camera_count) {
  std::vector<int> camera_ids;
  camera_ids.reserve(args.size());

  for (const std::string& arg : args) {
    std::size_t parsed_length = 0;
    const int camera_id = std::stoi(arg, &parsed_length);
    if (parsed_length != arg.size()) {
      throw std::invalid_argument("camera id contains trailing characters");
    }
    camera_ids.push_back(camera_id);
  }

  if (!camera_ids.empty()) {
    return camera_ids;
  }

  for (int i = 0; i < default_camera_count; ++i) {
    camera_ids.push_back(i);
  }

  return camera_ids;
}

GridShape compute_grid_shape(int tile_count) {
  if (tile_count <= 0) {
    return {.rows = 0, .columns = 0};
  }

  const int columns = tile_count == 1 ? 1 : 2;
  const int rows = static_cast<int>(std::ceil(tile_count / static_cast<double>(columns)));
  return {.rows = rows, .columns = columns};
}

}  // namespace cmake_demo
