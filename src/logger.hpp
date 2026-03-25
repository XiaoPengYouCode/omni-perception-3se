#pragma once

#include <chrono>
#include <cstdio>
#include <source_location>
#include <string_view>
#include <utility>

#include <fmt/chrono.h>
#include <fmt/color.h>

namespace cmake_demo {

namespace detail {

enum class LogLevel {
  kDebug,
  kInfo,
  kWarn,
  kError,
};

inline std::string_view level_name(LogLevel level) {
  switch (level) {
    case LogLevel::kDebug:
      return "DEBUG";
    case LogLevel::kInfo:
      return "INFO";
    case LogLevel::kWarn:
      return "WARN";
    case LogLevel::kError:
      return "ERROR";
  }

  return "INFO";
}

inline fmt::text_style level_style(LogLevel level) {
  switch (level) {
    case LogLevel::kDebug:
      return fmt::fg(fmt::color::steel_blue);
    case LogLevel::kInfo:
      return fmt::fg(fmt::color::light_green);
    case LogLevel::kWarn:
      return fmt::fg(fmt::color::golden_rod);
    case LogLevel::kError:
      return fmt::fg(fmt::color::crimson) | fmt::emphasis::bold;
  }

  return fmt::fg(fmt::color::white);
}

inline std::string_view short_file_name(std::string_view path) {
  const std::size_t pos = path.find_last_of("/\\");
  return pos == std::string_view::npos ? path : path.substr(pos + 1);
}

template <typename... Args>
void log_message(FILE* stream, LogLevel level, fmt::format_string<Args...> format,
                 const std::source_location& location, Args&&... args) {
  const auto now = std::chrono::floor<std::chrono::milliseconds>(
      std::chrono::system_clock::now());
  const std::string_view file_name = short_file_name(location.file_name());

  fmt::print(stream, fmt::fg(fmt::color::dark_gray), "{:%H:%M:%S.%e} | ", now);
  fmt::print(stream, level_style(level), "{:<5}", level_name(level));
  fmt::print(stream, fmt::fg(fmt::color::dark_gray), " | ");
  fmt::print(stream, fmt::fg(fmt::color::light_steel_blue), "{}:{}", file_name,
             location.line());
  fmt::print(stream, fmt::fg(fmt::color::dark_gray), " | ");
  fmt::print(stream, format, std::forward<Args>(args)...);
  fmt::print(stream, "\n");
}

}  // namespace detail

template <typename... Args>
void log_debug_impl(const std::source_location& location, fmt::format_string<Args...> format,
                    Args&&... args) {
  detail::log_message(stdout, detail::LogLevel::kDebug, format, location,
                      std::forward<Args>(args)...);
}

template <typename... Args>
void log_info_impl(const std::source_location& location, fmt::format_string<Args...> format,
                   Args&&... args) {
  detail::log_message(stdout, detail::LogLevel::kInfo, format, location,
                      std::forward<Args>(args)...);
}

template <typename... Args>
void log_warn_impl(const std::source_location& location, fmt::format_string<Args...> format,
                   Args&&... args) {
  detail::log_message(stderr, detail::LogLevel::kWarn, format, location,
                      std::forward<Args>(args)...);
}

template <typename... Args>
void log_error_impl(const std::source_location& location, fmt::format_string<Args...> format,
                    Args&&... args) {
  detail::log_message(stderr, detail::LogLevel::kError, format, location,
                      std::forward<Args>(args)...);
}

}  // namespace cmake_demo

#define log_debug(...) log_debug_impl(std::source_location::current(), __VA_ARGS__)
#define log_info(...) log_info_impl(std::source_location::current(), __VA_ARGS__)
#define log_warn(...) log_warn_impl(std::source_location::current(), __VA_ARGS__)
#define log_error(...) log_error_impl(std::source_location::current(), __VA_ARGS__)
