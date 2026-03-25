#include "core/app_config.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

TEST(ParseCameraIdsTest, UsesDefaultIdsWhenArgsEmpty) {
  const std::vector<int> camera_ids = op3::parse_camera_ids({});

  EXPECT_EQ(camera_ids, (std::vector<int>{0, 1, 2}));
}

TEST(ParseCameraIdsTest, ParsesExplicitIds) {
  const std::vector<int> camera_ids =
      op3::parse_camera_ids(std::vector<std::string>{"3", "7", "9"});

  EXPECT_EQ(camera_ids, (std::vector<int>{3, 7, 9}));
}

TEST(ParseCameraIdsTest, RejectsMalformedIds) {
  EXPECT_THROW(op3::parse_camera_ids(std::vector<std::string>{"1x"}),
               std::invalid_argument);
}

TEST(ComputeGridShapeTest, BuildsSingleTileGrid) {
  const op3::GridShape shape = op3::compute_grid_shape(1);

  EXPECT_EQ(shape.rows, 1);
  EXPECT_EQ(shape.columns, 1);
}

TEST(ComputeGridShapeTest, BuildsTwoColumnGridForThreeTiles) {
  const op3::GridShape shape = op3::compute_grid_shape(3);

  EXPECT_EQ(shape.rows, 2);
  EXPECT_EQ(shape.columns, 2);
}

TEST(ComputeGridShapeTest, ReturnsZeroShapeForEmptyGrid) {
  const op3::GridShape shape = op3::compute_grid_shape(0);

  EXPECT_EQ(shape.rows, 0);
  EXPECT_EQ(shape.columns, 0);
}

}  // namespace
