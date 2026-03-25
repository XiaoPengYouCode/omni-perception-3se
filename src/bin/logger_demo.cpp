/** Simple executable that exercises the logger formatting helpers. */
#include "infra/logger.hpp"

int main() {
  op3::log_debug("Debug message");
  op3::log_info("Info message");
  op3::log_warn("Warn message");
  op3::log_error("Error message");
  return 0;
}
