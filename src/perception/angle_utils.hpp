#pragma once

#include <string>

#include "perception/types.hpp"

namespace op3 {

/**
 * Returns the nominal robot-frame heading for a given camera mount.
 */
double camera_base_angle(CameraPosition camera);

/**
 * Returns the default model used by the built-in four-camera rig.
 */
CameraModel make_default_camera_model(CameraPosition camera, int image_width = 960,
                                      int image_height = 540);

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
 * Estimates person range from a monocular bounding box size using both width and height.
 */
double estimate_person_range(const cv::Rect& bbox, const CameraModel& model);

/**
 * Computes the horizontal-FOV-based focal length used by the simple pinhole model.
 */
double focal_length_pixels(int image_width, double horizontal_fov_degrees);

/**
 * Projects the assumed person height into image pixels at the given range.
 */
double project_person_height_pixels(double range_m, const CameraModel& model);

/**
 * Projects the assumed person width into image pixels at the given range.
 */
double project_person_width_pixels(double range_m, const CameraModel& model);

/**
 * Converts body-frame polar coordinates into planar x/y coordinates.
 */
cv::Point2d body_frame_point_from_polar(double range_m, double angle_degrees);

/**
 * Rotates a planar vector by the given yaw angle.
 */
cv::Point2d rotate_point(double x_m, double y_m, double yaw_degrees);

/**
 * Converts a body-frame point into a world-frame point using the robot pose.
 */
cv::Point2d body_to_world_point(double x_m, double y_m, const RobotPose& robot_pose);

/**
 * Converts a world-frame point into a body-frame point using the robot pose.
 */
cv::Point2d world_to_body_point(double x_m, double y_m, const RobotPose& robot_pose);

/**
 * Computes a robot-frame angle from a planar x/y point.
 */
double body_frame_angle_from_point(double x_m, double y_m);

/**
 * Computes Euclidean range from a planar x/y point.
 */
double body_frame_range_from_point(double x_m, double y_m);

/**
 * Converts a camera enum into a stable JSON-friendly string.
 */
std::string camera_position_to_string(CameraPosition camera);

} // namespace op3
