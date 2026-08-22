// pybind11 bindings: exposes the C++ control-loop core to Python as the
// `adas_core` extension module.
//
// Interop notes (the things worth being able to explain):
//
//  * Every function here takes and returns plain doubles. pybind11 converts
//    Python float <-> C++ double by value, so no ownership is transferred and
//    there is nothing to free -- the C++ side allocates nothing per call.
//  * py::arg(...) declarations give the bound functions the SAME keyword
//    argument names and defaults as the Python reference implementations, so
//    they are drop-in replacements at every existing call site.
//  * aeb_controller returns a C++ `enum class` internally but is bound to
//    return a std::string, which pybind11 converts to a Python str. That keeps
//    the C++ side type-safe without changing what Python callers receive.
//  * brake_response_latency throws std::invalid_argument, which pybind11's
//    built-in exception translation surfaces in Python as ValueError -- the
//    same exception type the pure-Python version raises.
//  * The GIL is held during these calls. They are nanosecond-scale pure
//    arithmetic, so releasing it would cost more than it saves; that is a
//    deliberate choice, not an oversight.

#include <pybind11/pybind11.h>

#include <string>

#include "adas_core/controllers.hpp"
#include "adas_core/metrics.hpp"

namespace py = pybind11;

PYBIND11_MODULE(adas_core, m) {
    m.doc() =
        "C++ implementation of the ADAS validation harness control-loop core "
        "(per-timestep metrics and controllers), exposed via pybind11. "
        "Numerically equivalent to the pure-Python reference in metrics/ and "
        "controllers/.";

    // ---- metrics -----------------------------------------------------

    m.def("time_to_collision",
          &adas_core::time_to_collision,
          py::arg("ego_speed"),
          py::arg("lead_speed"),
          py::arg("gap_distance"),
          "Seconds until collision assuming constant speeds; inf if not closing, "
          "0.0 if the gap has already closed.");

    m.def("lane_deviation",
          &adas_core::lane_deviation,
          py::arg("lateral_offset_from_center"),
          "Absolute lateral deviation from lane center, in meters.");

    m.def("following_distance_error",
          &adas_core::following_distance_error,
          py::arg("actual_gap"),
          py::arg("target_gap"),
          "Signed following-distance error in meters (positive = farther than target).");

    m.def("brake_response_latency",
          &adas_core::brake_response_latency,
          py::arg("event_timestamp"),
          py::arg("brake_onset_timestamp"),
          "Seconds between the triggering event and brake onset. Raises ValueError "
          "if brake onset precedes the event.");

    // ---- controllers -------------------------------------------------

    m.def("acc_controller",
          &adas_core::acc_controller,
          py::arg("ego_speed"),
          py::arg("lead_speed"),
          py::arg("gap"),
          py::arg("target_gap") = 2.0,
          py::arg("time_gap") = 1.5,
          py::arg("min_accel") = -4.0,
          py::arg("max_accel") = 2.0,
          "Proportional gap controller; commanded acceleration in m/s^2.");

    m.def("lka_controller",
          &adas_core::lka_controller,
          py::arg("lateral_offset"),
          py::arg("heading_error"),
          py::arg("max_steer") = 0.5,
          "PD lane-centering controller; commanded steering angle in radians.");

    m.def(
        "aeb_controller",
        [](double ttc, double warning_ttc, double brake_ttc) -> std::string {
            return adas_core::to_string(adas_core::aeb_controller(ttc, warning_ttc, brake_ttc));
        },
        py::arg("ttc"),
        py::arg("warning_ttc") = 2.0,
        py::arg("brake_ttc") = 1.8,
        "Staged TTC braking; returns 'normal', 'warning' or 'full_brake'.");
}
