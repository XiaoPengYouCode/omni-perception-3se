#pragma once

#include <string>
#include <vector>

namespace cmake_demo {

struct GridShape {
  int rows;
  int columns;
};

std::vector<int> parse_camera_ids(const std::vector<std::string>& args,
                                  int default_camera_count = 3);

GridShape compute_grid_shape(int tile_count);

}  // namespace cmake_demo
