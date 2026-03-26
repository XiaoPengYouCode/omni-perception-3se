#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace op3 {

/**
 * Returns the scene-specialized labels used by the current simulator setup.
 */
const std::vector<std::string>& known_scene_labels();

/**
 * Encodes a scene label into a dark grayscale value that survives the mock detector path.
 */
int scene_label_grayscale(std::string_view label);

/**
 * Decodes the closest scene label from a grayscale signature.
 */
std::string decode_scene_label_from_grayscale(double grayscale_value);

} // namespace op3
