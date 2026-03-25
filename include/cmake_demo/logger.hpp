#pragma once

#include <cstdio>
#include <utility>

#include <fmt/core.h>

namespace cmake_demo {

template <typename... Args>
void log_info(fmt::format_string<Args...> format, Args&&... args) {
  fmt::print("[INFO] ");
  fmt::print(format, std::forward<Args>(args)...);
  fmt::print("\n");
}

template <typename... Args>
void log_warn(fmt::format_string<Args...> format, Args&&... args) {
  fmt::print(stderr, "[WARN] ");
  fmt::print(stderr, format, std::forward<Args>(args)...);
  fmt::print(stderr, "\n");
}

template <typename... Args>
void log_error(fmt::format_string<Args...> format, Args&&... args) {
  fmt::print(stderr, "[ERROR] ");
  fmt::print(stderr, format, std::forward<Args>(args)...);
  fmt::print(stderr, "\n");
}

}  // namespace cmake_demo
