#pragma once

#include <string>
#include <vector>

namespace op3 {

/**
 * Describes a simple row/column tile layout.
 */
struct GridShape {
  int rows;
  int columns;
};

/**
 * Parses explicit camera ids from CLI args, or returns a default contiguous range.
 *
 * @param args Raw CLI arguments that should contain integer camera ids.
 * @param default_camera_count Number of default ids to return when args are empty.
 * @return Parsed or synthesized camera ids.
 */
std::vector<int> parse_camera_ids(const std::vector<std::string>& args,
                                  int default_camera_count = 3);

/**
 * Computes a compact display grid for a given number of tiles.
 *
 * @param tile_count Number of tiles that need to fit in the grid.
 * @return Grid shape with zero rows/columns when tile_count is not positive.
 */
GridShape compute_grid_shape(int tile_count);

} // namespace op3
