cmake_minimum_required(VERSION 3.20)

project(omni-perception-3se VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(fmt CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED COMPONENTS core highgui imgproc videoio)

add_library(omni_perception_3se_core
  src/core/app_config.cpp
  src/perception/angle_utils.cpp
  src/perception/camera_worker.cpp
  src/perception/detector_factory.cpp
  src/perception/fusion_tracker.cpp
  src/perception/json_output.cpp
  src/perception/mock_person_detector.cpp
  src/perception/scene_labels.cpp
  src/perception/state_publisher.cpp
  src/perception/yolo26_nano_detector.cpp
  src/simulation/simulation_engine.cpp
)

target_include_directories(omni_perception_3se_core PUBLIC src)

target_link_libraries(omni_perception_3se_core
  PUBLIC
    fmt::fmt
    opencv_core
    opencv_imgproc
)

add_executable(omni-perception-3se src/main.cpp)
add_executable(logger_demo src/bin/logger_demo.cpp)

target_link_libraries(omni-perception-3se
  PRIVATE
    omni_perception_3se_core
    fmt::fmt
    opencv_highgui
)

target_link_libraries(logger_demo
  PRIVATE
    omni_perception_3se_core
    fmt::fmt
)

if(BUILD_TESTING)
  find_package(GTest CONFIG REQUIRED)

  add_executable(omni_perception_3se_tests
    tests/app_config_test.cpp
    tests/pipeline_test.cpp
    tests/simulation_test.cpp
  )

  target_link_libraries(omni_perception_3se_tests
    PRIVATE
      omni_perception_3se_core
      GTest::gtest_main
  )

  include(GoogleTest)
  gtest_discover_tests(omni_perception_3se_tests)
endif()
