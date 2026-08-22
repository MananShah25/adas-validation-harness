#include "adas_core/metrics.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace adas_core {

double time_to_collision(double ego_speed, double lead_speed, double gap_distance) noexcept {
    const double closing_speed = ego_speed - lead_speed;

    // Not closing -> no collision to predict. std::numeric_limits<double>
    // infinity is what pybind11 hands back to Python as math.inf, so the
    // Python-side `math.isinf(...)` checks in the harness keep working.
    if (closing_speed <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    // Gap already closed; report 0 rather than a negative or divide-by-zero.
    if (gap_distance <= 0.0) {
        return 0.0;
    }
    return gap_distance / closing_speed;
}

double lane_deviation(double lateral_offset_from_center) noexcept {
    return std::fabs(lateral_offset_from_center);
}

double following_distance_error(double actual_gap, double target_gap) noexcept {
    return actual_gap - target_gap;
}

double brake_response_latency(double event_timestamp, double brake_onset_timestamp) {
    const double latency = brake_onset_timestamp - event_timestamp;
    if (latency < 0.0) {
        // Braking before the triggering event is a caller ordering bug, not a
        // fast vehicle. Throwing std::invalid_argument here surfaces in Python
        // as ValueError via the pybind11 exception translation table, matching
        // the reference implementation's contract exactly.
        throw std::invalid_argument(
            "brake_onset_timestamp (" + std::to_string(brake_onset_timestamp) +
            ") is before event_timestamp (" + std::to_string(event_timestamp) + ")");
    }
    return latency;
}

}  // namespace adas_core
