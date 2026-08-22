#include "adas_core/controllers.hpp"

#include <algorithm>

namespace adas_core {
namespace {

// Python's max(lo, min(hi, x)) written once. Deliberately not std::clamp:
// std::clamp is undefined behaviour when hi < lo, whereas this reproduces the
// reference implementation's exact behaviour for every input including that
// degenerate case.
constexpr double clip(double value, double lo, double hi) noexcept {
    return std::max(lo, std::min(hi, value));
}

}  // namespace

const char* to_string(AebState state) noexcept {
    switch (state) {
        case AebState::FullBrake:
            return "full_brake";
        case AebState::Warning:
            return "warning";
        case AebState::Normal:
            break;
    }
    return "normal";
}

double acc_controller(double ego_speed,
                      double lead_speed,
                      double gap,
                      double target_gap,
                      double time_gap,
                      double min_accel,
                      double max_accel) noexcept {
    const double desired_gap = target_gap + time_gap * ego_speed;
    const double gap_error = gap - desired_gap;
    const double speed_error = lead_speed - ego_speed;
    const double accel = 0.5 * gap_error + 0.3 * speed_error;
    return clip(accel, min_accel, max_accel);
}

double lka_controller(double lateral_offset, double heading_error, double max_steer) noexcept {
    const double steering = -0.3 * lateral_offset - 0.5 * heading_error;
    return clip(steering, -max_steer, max_steer);
}

AebState aeb_controller(double ttc, double warning_ttc, double brake_ttc) noexcept {
    // Strict less-than at both thresholds, matching the Python reference:
    // a TTC exactly equal to a threshold does NOT trigger that stage.
    if (ttc < brake_ttc) {
        return AebState::FullBrake;
    }
    if (ttc < warning_ttc) {
        return AebState::Warning;
    }
    return AebState::Normal;
}

}  // namespace adas_core
