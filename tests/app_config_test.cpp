#include "app_config.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

TEST(ParseCameraIdsTest, UsesDefaultIdsWhenArgsEmpty) {
  const std::vector<int> camera_ids = cmake_demo::parse_camera_ids({});

  EXPECT_EQ(camera_ids, (std::vector<int>{0, 1, 2}));
}

TEST(ParseCameraIdsTest, ParsesExplicitIds) {
  const std::vector<int> camera_ids =
      cmake_demo::parse_camera_ids(std::vector<std::string>{"3", "7", "9"});

  EXPECT_EQ(camera_ids, (std::vector<int>{3, 7, 9}));
}

TEST(ParseCameraIdsTest, RejectsMalformedIds) {
  EXPECT_THROW(cmake_demo::parse_camera_ids(std::vector<std::string>{"1x"}),
               std::invalid_argument);
}

TEST(ComputeGridShapeTest, BuildsSingleTileGrid) {
  const cmake_demo::GridShape shape = cmake_demo::compute_grid_shape(1);

  EXPECT_EQ(shape.rows, 1);
  EXPECT_EQ(shape.columns, 1);
}

TEST(ComputeGridShapeTest, BuildsTwoColumnGridForThreeTiles) {
  const cmake_demo::GridShape shape = cmake_demo::compute_grid_shape(3);

  EXPECT_EQ(shape.rows, 2);
  EXPECT_EQ(shape.columns, 2);
}

TEST(ComputeGridShapeTest, ReturnsZeroShapeForEmptyGrid) {
  const cmake_demo::GridShape shape = cmake_demo::compute_grid_shape(0);

  EXPECT_EQ(shape.rows, 0);
  EXPECT_EQ(shape.columns, 0);
}

}  // namespace
