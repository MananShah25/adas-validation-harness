#pragma once

#include <string>

// The three ADAS controllers -- the systems under test -- ported from
// controllers/{acc,lka,aeb}.py. Kept intentionally simple and rule-based so
// they remain defensible line by line; the C++ port changes the language, not
// the algorithm, so results must match the Python reference exactly.

namespace adas_core {

// Discrete AEB output. An enum class rather than a bare string keeps the state
// machine type-safe and non-convertible on the C++ side; the pybind11 layer
// converts to the same "normal"/"warning"/"full_brake" strings the Python
// harness already consumes, so callers see no difference.
enum class AebState { Normal, Warning, FullBrake };

// String form of an AebState, matching the Python controller's return values.
const char* to_string(AebState state) noexcept;

// Proportional gap controller for Adaptive Cruise Control.
// Returns commanded longitudinal acceleration in m/s^2, clipped to
// [min_accel, max_accel].
double acc_controller(double ego_speed,
                      double lead_speed,
                      double gap,
                      double target_gap = 2.0,
                      double time_gap = 1.5,
                      double min_accel = -4.0,
                      double max_accel = 2.0) noexcept;

// Proportional-derivative lane centering for Lane Keeping Assist.
// Returns commanded steering angle in radians, clipped to +/- max_steer.
double lka_controller(double lateral_offset,
                      double heading_error,
                      double max_steer = 0.5) noexcept;

// Staged time-to-collision braking for Automatic Emergency Braking.
// Thresholds default to the NHTSA-sourced values documented in
// schemas/test_procedure.py: 2.0s warning, 1.8s brake.
AebState aeb_controller(double ttc,
                        double warning_ttc = 2.0,
                        double brake_ttc = 1.8) noexcept;

}  // namespace adas_core
