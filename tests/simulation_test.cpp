#include "simulation/simulation_engine.hpp"

#include <gtest/gtest.h>

namespace {

TEST(SimulationEngineTest, RendersFourCameraFrames) {
  op3::SimulationEngine simulation(op3::SimulationEngine::default_config());

  const std::vector<op3::FrameInput> frames = simulation.render_frames();

  ASSERT_EQ(frames.size(), 4U);
  EXPECT_EQ(frames[0].camera, op3::CameraPosition::kLeftFront);
  EXPECT_EQ(frames[1].camera, op3::CameraPosition::kRightFront);
  EXPECT_EQ(frames[2].camera, op3::CameraPosition::kLeftRear);
  EXPECT_EQ(frames[3].camera, op3::CameraPosition::kRightRear);
  for (const op3::FrameInput& frame : frames) {
    EXPECT_FALSE(frame.image.empty());
  }
}

TEST(SimulationEngineTest, StartsRobotNearMapCenter) {
  const op3::SimulationConfig config = op3::SimulationEngine::default_config();
  op3::SimulationEngine simulation(config);

  const op3::RobotState robot = simulation.robot_state();

  EXPECT_DOUBLE_EQ(robot.position.x, config.map_size.width * 0.5);
  EXPECT_DOUBLE_EQ(robot.position.y, config.map_size.height * 0.5);
}

TEST(SimulationEngineTest, MovesRobotForwardWithControlInput) {
  op3::SimulationEngine simulation(op3::SimulationEngine::default_config());
  const op3::RobotState before = simulation.robot_state();

  op3::SimulationControls controls;
  controls.move_forward = true;
  simulation.update(0.5, controls);

  const op3::RobotState after = simulation.robot_state();
  EXPECT_GT(after.position.x, before.position.x);
  EXPECT_DOUBLE_EQ(after.position.y, before.position.y);
}

TEST(SimulationEngineTest, AdvancesPedestriansAlongWaypoints) {
  op3::SimulationEngine simulation(op3::SimulationEngine::default_config());
  const std::vector<op3::PedestrianState> before = simulation.pedestrian_states();

  simulation.update(1.0, {});

  const std::vector<op3::PedestrianState> after = simulation.pedestrian_states();
  ASSERT_EQ(before.size(), after.size());
  const bool moved = after.front().position.x != before.front().position.x ||
                     after.front().position.y != before.front().position.y;
  EXPECT_TRUE(moved);
}

} // namespace
