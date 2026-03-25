cmake_minimum_required(VERSION 3.20)

project(cmake_demo VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(fmt CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED COMPONENTS core highgui imgproc videoio)

add_library(cmake_demo_core
  src/app_config.cpp
)

target_include_directories(cmake_demo_core PUBLIC include)

add_executable(cmake_demo src/main.cpp)

target_link_libraries(cmake_demo
  PRIVATE
    cmake_demo_core
    fmt::fmt
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_videoio
)

if(BUILD_TESTING)
  find_package(GTest CONFIG REQUIRED)

  add_executable(cmake_demo_tests
    tests/app_config_test.cpp
  )

  target_link_libraries(cmake_demo_tests
    PRIVATE
      cmake_demo_core
      GTest::gtest_main
  )

  include(GoogleTest)
  gtest_discover_tests(cmake_demo_tests)
endif()
