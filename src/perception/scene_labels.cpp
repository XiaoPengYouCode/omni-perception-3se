#include "perception/scene_labels.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace op3 {

namespace {

struct LabelSignature {
  std::string_view label;
  int grayscale;
};

constexpr std::array<LabelSignature, 4> kLabelSignatures = {{
    {"Hero", 20},
    {"Engineer", 35},
    {"infantry", 50},
    {"sentry", 65},
}};

} // namespace

const std::vector<std::string>& known_scene_labels() {
  static const std::vector<std::string> labels = {"Hero", "Engineer", "infantry", "sentry"};
  return labels;
}

int scene_label_grayscale(std::string_view label) {
  for (const LabelSignature& signature : kLabelSignatures) {
    if (signature.label == label) {
      return signature.grayscale;
    }
  }

  return 20;
}

std::string decode_scene_label_from_grayscale(double grayscale_value) {
  const LabelSignature* best_match = &kLabelSignatures.front();
  double best_distance = std::numeric_limits<double>::max();
  for (const LabelSignature& signature : kLabelSignatures) {
    const double distance = std::abs(grayscale_value - static_cast<double>(signature.grayscale));
    if (distance < best_distance) {
      best_distance = distance;
      best_match = &signature;
    }
  }

  return std::string(best_match->label);
}

} // namespace op3
