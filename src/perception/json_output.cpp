#include "perception/json_output.hpp"

#include <chrono>

#include <fmt/format.h>

#include "perception/angle_utils.hpp"

namespace op3 {

namespace {

/**
 * Escapes a string for safe insertion into a JSON string literal.
 */
std::string json_escape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());

  for (const char ch : value) {
    switch (ch) {
    case '\\':
      escaped += "\\\\";
      break;
    case '"':
      escaped += "\\\"";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      escaped.push_back(ch);
      break;
    }
  }

  return escaped;
}

/**
 * Converts steady_clock timestamps into integer milliseconds for JSON output.
 */
std::int64_t to_epoch_milliseconds(std::chrono::steady_clock::time_point timestamp) {
  return static_cast<std::int64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(timestamp.time_since_epoch()).count());
}

} // namespace

std::string to_json(const PipelineOutput& output) {
  std::string json = fmt::format("{{\"sequence_id\":{},\"timestamp_ms\":{},\"person\":[",
                                 output.sequence_id, output.timestamp_ms);

  // Serialize the current fused tracks as a flat array of state objects.
  for (std::size_t i = 0; i < output.person.size(); ++i) {
    const TrackedPerson& report = output.person[i];
    if (i > 0) {
      json += ",";
    }

    std::string sources = "[";
    for (std::size_t source_index = 0; source_index < report.sources.size(); ++source_index) {
      if (source_index > 0) {
        sources += ",";
      }
      sources += fmt::format("\"{}\"", camera_position_to_string(report.sources[source_index]));
    }
    sources += "]";

    json += fmt::format(
        "{{\"track_id\":\"{}\",\"angle\":{:.1f},\"angle_velocity\":{:.1f},\"confidence\":{:.2f},"
        "\"sources\":{},\"last_update_ms\":{}}}",
        json_escape(report.track_id), report.angle, report.angle_velocity, report.confidence,
        sources, to_epoch_milliseconds(report.last_update));
  }

  json += "]}";
  return json;
}

} // namespace op3
