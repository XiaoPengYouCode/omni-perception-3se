#pragma once

#include <vector>

#include "perception/types.hpp"

namespace op3 {

/**
 * Builds four synthetic frames that exercise the end-to-end pipeline without hardware.
 */
std::vector<FrameInput> make_demo_frames(int image_width = 960, int image_height = 540);

} // namespace op3
