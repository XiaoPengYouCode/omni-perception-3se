#include "logger.hpp"

int main() {
  cmake_demo::log_debug("Debug message");
  cmake_demo::log_info("Info message");
  cmake_demo::log_warn("Warn message");
  cmake_demo::log_error("Error message");
  return 0;
}
