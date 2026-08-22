# C++ control-loop core

The per-timestep control loop of the ADAS validation harness — the ADAS
controllers and the safety metrics — implemented in C++17 and driven from the
existing Python scenarios through pybind11.

```bash
./cpp/build.sh        # configure, build, run GoogleTest + Python parity tests
```

---

## What is in C++ and what is not

Being precise about this matters more than the line count.

**In C++** — the functions that execute once per simulation timestep, are pure
(scalar in, scalar out, no allocation, no I/O), and carry the safety logic:

| C++ | Python reference |
|---|---|
| `time_to_collision` | `metrics/ttc_lane_brake.py` |
| `lane_deviation` | `metrics/ttc_lane_brake.py` |
| `following_distance_error` | `metrics/ttc_lane_brake.py` |
| `brake_response_latency` | `metrics/ttc_lane_brake.py` |
| `acc_controller` | `controllers/acc.py` |
| `lka_controller` | `controllers/lka.py` |
| `aeb_controller` | `controllers/aeb.py` |

**Not in C++** — the vehicle physics (`highway-env`), the scenario definitions,
the OBD-II ingestion, YOLOv8 inference, and the reporting layer. Those stay in
Python. The accurate claim is *"the control-loop core is C++, driven from
Python"* — not "the simulation is in C++".

## Layout

```
cpp/
├── CMakeLists.txt              build: static lib + pybind11 module + gtest
├── build.sh                    one-command configure/build/test
├── include/adas_core/
│   ├── metrics.hpp             TTC, deviation, gap error, brake latency
│   └── controllers.hpp         ACC, LKA, AEB (+ AebState enum class)
├── src/
│   ├── metrics.cpp
│   ├── controllers.cpp
│   └── bindings.cpp            the pybind11 interop layer
└── tests/
    ├── test_metrics.cpp        GoogleTest
    └── test_controllers.cpp    GoogleTest
```

The algorithms live in `adas_core_lib`, a static library with **no Python
dependency at all**. The pybind11 module links against it. That separation means
the GoogleTest suite exercises exactly the same objects Python calls, and the
core could be reused in a non-Python (e.g. embedded) context.

## Memory model and interop

Worth being able to explain out loud:

- **Everything crosses the boundary by value.** Every bound function takes and
  returns `double`. pybind11 converts Python `float` ↔ C++ `double` by copy, so
  no ownership transfers, nothing is shared, and there is nothing to free. The
  C++ side performs **zero heap allocations** per call.
- **No raw pointers, no manual `new`/`delete`** anywhere in the core. The one
  non-trivial object is the exception message `std::string`, which is RAII-managed
  and destroyed as the exception is translated.
- **Exception translation.** `brake_response_latency` throws
  `std::invalid_argument`; pybind11's built-in translation surfaces that in
  Python as `ValueError` — the same type the pure-Python version raises, so the
  existing contract is preserved.
- **Type safety at the seam.** `aeb_controller` returns a C++ `enum class
  AebState` internally (non-convertible, exhaustively switched) but is bound to
  return `std::string`, so Python still receives `"normal"` / `"warning"` /
  `"full_brake"`. Strong typing inside, unchanged interface outside.
- **Keyword arguments and defaults are declared** via `py::arg(...)` to match the
  Python signatures exactly, which is what makes the C++ functions drop-in
  replacements at existing call sites.
- **The GIL is held** during these calls. They are nanosecond-scale arithmetic;
  releasing and reacquiring the GIL would cost far more than the work itself.
  That is a deliberate choice, not an oversight.

## The floating-point bug worth talking about

The parity test caught a real defect that inspection would have missed.

Initially the C++ `lka_controller` disagreed with Python in the last bit:

```
lka(0.3, 0.1)   C++ = -0.13999999999999999
                Py  = -0.14000000000000001
```

**Cause:** on ARM64, clang defaults to floating-point *contraction* — it fuses
`-0.3 * offset - 0.5 * heading` into a single FMA instruction, which rounds
**once**. CPython evaluates each multiply and the subtraction as separate
IEEE-754 operations, rounding **at each step**. The FMA answer is marginally
*more* accurate; it is simply not the same answer.

**Why it matters here:** this harness compares measured values against fixed
thresholds (`0.3 m` lane deviation, `1.8 s` brake TTC). A last-ulp difference can
flip a borderline PASS to a FAIL, which in a validation tool is a correctness
bug, not a rounding curiosity.

**Fix:** `-ffp-contract=off` on the core library (see the comment in
`CMakeLists.txt`). Determinism against the reference beats marginally better
accuracy for this use case.

## Verification

Three independent layers:

1. **GoogleTest — 32 tests.** Unit tests on the C++ side alone, including
   infinity handling, clipping limits, and the exclusive threshold boundaries
   (a TTC exactly equal to a threshold must *not* trigger that stage).
2. **Grid parity — `tests/test_cpp_python_parity.py`.** Bit-exact comparison
   against the Python reference across ~1,000 TTC input combinations, ~1,000 ACC
   combinations, every LKA offset/heading pair, and every AEB threshold
   boundary. Also asserts both sides reject the same bad inputs with the same
   exception type.
3. **Scenario equivalence — `tests/test_cpp_scenario_equivalence.py`.** Runs all
   29 *unmodified* Track 1 scenarios with the C++ core swapped in at runtime and
   asserts every observed metric and pass/fail verdict is identical, and that the
   headline pass rates (ACC 6/10, LKA 8/10, AEB 8/9) are unchanged.

Layer 3 is the one that substantiates the claim: the Python harness above the
core is genuinely untouched.

## Performance — the honest version

Measured on Apple Silicon, 200k calls each:

| Function | Python | C++ via pybind11 | Ratio |
|---|---|---|---|
| `time_to_collision` | ~112 ns/call | ~75 ns/call | ~1.5× |
| `acc_controller` | ~268 ns/call | ~120 ns/call | ~2.2× |

**Do not oversell this.** At this granularity the timing is dominated by the
pybind11 boundary crossing, not by the arithmetic — the actual math is a handful
of nanoseconds in both languages. Calling C++ one scalar at a time from Python
spends most of the saving on the FFI overhead.

The real speedup would come from moving the *entire* stepping loop into C++ so
the boundary is crossed once per scenario instead of once per timestep. That is
the honest next step, and it is deliberately **not** claimed here.

The defensible reasons this port is worthwhile as it stands: a strongly-typed,
separately-testable core; a reusable library with no Python dependency; and a
demonstrated bit-exact equivalence discipline between two implementations.

## Build details

- **C++17**, `-Wall -Wextra -Wpedantic`, clean.
- **CMake ≥ 3.18**; GoogleTest fetched at configure time via `FetchContent`
  (pinned to v1.15.2), so there is no vendored dependency in the repo.
- `POSITION_INDEPENDENT_CODE ON` on the static library, because it is linked
  into a shared object (the Python extension).
- The built `.so` is written to the repo root so `import adas_core` works
  without installing or setting `PYTHONPATH`.
- Build artifacts (`cpp/build/`, `*.so`) are gitignored; only sources are
  tracked.

Both Python test files `importorskip` the extension, so the full suite still
passes on a machine with no C++ toolchain — the C++ path is additive, never a
prerequisite.
