#pragma once

// Core ADAS metrics -- the pure, per-timestep numeric functions ported from
// metrics/ttc_lane_brake.py. These run once per simulation step and have no
// dependency on highway-env, numpy, or any I/O, which is what makes them a
// clean, deterministic target for a C++ implementation driven from Python.
//
// Every function here is a leaf: scalar doubles in, a scalar double out, no
// allocation, no shared state. That is deliberate -- it keeps the language
// interop trivial (Python float <-> C++ double) and the results bit-for-bit
// reproducible against the Python reference.

namespace adas_core {

// Seconds until ego collides with the lead vehicle assuming constant speeds.
// Returns +infinity if the gap is not closing (no collision to predict), and
// 0.0 if the gap has already closed. Mirrors math.inf semantics on the Python
// side.
double time_to_collision(double ego_speed, double lead_speed, double gap_distance) noexcept;

// Absolute lateral deviation from lane center, in meters.
double lane_deviation(double lateral_offset_from_center) noexcept;

// Signed error between actual and target following distance, in meters.
// Positive = farther than target, negative = closer.
double following_distance_error(double actual_gap, double target_gap) noexcept;

// Seconds between the triggering event and brake onset.
// Throws std::invalid_argument if brake onset precedes the event -- pybind11
// maps that to a Python ValueError, preserving the reference contract.
double brake_response_latency(double event_timestamp, double brake_onset_timestamp);

}  // namespace adas_core
