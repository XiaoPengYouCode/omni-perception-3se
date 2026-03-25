#pragma once

#include <string>

#include "perception/types.hpp"

namespace op3 {

/**
 * Serializes a fused tracker snapshot into the JSON contract exposed by the app.
 */
std::string to_json(const PipelineOutput& output);

} // namespace op3
