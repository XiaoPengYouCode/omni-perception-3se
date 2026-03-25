#pragma once

#include <string>

#include "perception/types.hpp"

namespace op3 {

/**
 * Returns the nominal robot-frame heading for a given camera mount.
 */
double camera_base_angle(CameraPosition camera);

/**
 * Wraps an angle into the [-180, 180) range.
 */
double normalize_angle(double angle_degrees);

/**
 * Converts a detected bounding box into a robot-frame angle estimate.
 *
 * @param camera Camera that produced the observation.
 * @param bbox Detection bounding box in image coordinates.
 * @param image_width Width of the source image in pixels.
 * @param horizontal_fov_degrees Assumed horizontal field of view of the camera.
 * @return Estimated body-frame angle of the detected person.
 */
double compute_person_angle(CameraPosition camera, const cv::Rect& bbox, int image_width,
                            double horizontal_fov_degrees = 90.0);

/**
 * Converts a camera enum into a stable JSON-friendly string.
 */
std::string camera_position_to_string(CameraPosition camera);

}  // namespace op3
