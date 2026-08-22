# ADAS Validation Harness

A validation framework for three ADAS features — Adaptive Cruise Control (ACC),
Lane Keeping Assist (LKA), and Automatic Emergency Braking (AEB) — with two
independent data tracks feeding **one shared scoring and reporting core**:

- **Track 1 — Simulation.** Scripted [`highway-env`](https://github.com/Farama-Foundation/HighwayEnv)
  scenarios exercise rule-based ACC/LKA/AEB controllers and score them against
  written, sourced acceptance criteria.
- **Track 2 — Real-vehicle software backbone.** The ingestion → vision → fusion
  pipeline that will consume real OBD-II and dashcam data once hardware is
  attached, built and tested today against mock data, a real public OBD-II
  dataset, and real object detection.

Both tracks write into the **same** test-procedure schema and the **same**
defect-log format, so a threshold means the same thing regardless of whether
the numbers came from a physics simulation or a real car.

The per-timestep **control-loop core is additionally implemented in C++17** and
driven from the same Python scenarios via pybind11, with bit-exact parity proven
against the Python reference (see [`cpp/`](cpp/README.md)).

Everything runs on a MacBook with no external hardware. **143 Python tests + 32
GoogleTest tests pass.**

---

## Why this is a validation harness, not a driving demo

The environment (or the sensor) validates nothing on its own. What makes this a
*harness* is that every run is scored against `pass_criteria` and every failure
is logged as a defect with reproduction steps — the same "steps / expected /
observed" shape used in professional test writeups. The controllers are the
**system under test**, deliberately simple and rule-based so they are defensible
line by line; the point is a correct, testable baseline, not a clever algorithm.

A useful consequence: **not everything passes, and that's the design working.**

| Feature | Scenarios | Passing | The failures are… |
|---|---|---|---|
| ACC | 10 | 6 | braking beyond the controller's −4 m/s² authority, aggressive stop-and-go, two cut-ins a simple P-controller doesn't settle |
| LKA | 10 | 8 | a large initial offset and a tight curve — no curvature feedforward |
| AEB | 9 | 8 | an already-critical starting gap, unrecoverable at any braking authority |

Each failure was verified against raw trajectory data, not inferred from a
boolean. A 100% pass rate would have meant the scenarios weren't testing
anything.

---

## Architecture

An hourglass: two very different data sources narrow into one shared set of
types and metrics, then widen back out into one report format.

```
  SOURCES            highway-env        OBDReader          VehicleDetector
                     (scripted)      (mock/replay/live)   (YOLOv8n + distance)
                          |                 |                     |
  SYSTEMS UNDER TEST  acc_controller   lka_controller       aeb_controller
                          |                 |                     |
                          v                 v                     v
  SHARED CORE  ......  metrics/  +  TestProcedure/TestResult  (the narrow waist)
                          |
                          v
  OUTPUT              DefectLog (CSV)  +  ReportBuilder (Markdown / PDF + plots)
```

The shared core (`schemas/`, `metrics/`, `reporting/`) is built first because
everything depends on it. The two tracks never diverge from it — a dedicated
convergence check (`tests/test_convergence.py`) asserts both produce the
identical report and defect-log format, structurally *and* by type identity.

### Repository layout

```
schemas/test_procedure.py     TestProcedure + TestResult, sourced pass_criteria
metrics/ttc_lane_brake.py     time_to_collision, lane_deviation, following-distance, brake latency
reporting/defect_log.py       append-only CSV defect log + Markdown render
reporting/report_builder.py   Markdown and paginated-PDF reports with matplotlib plots
controllers/                  acc.py, lka.py, aeb.py  (the systems under test)
scenarios/                    acc/lka/aeb scenario suites + track1_suite.py
ingestion/obd_reader.py       OBDReader: mock | replay | live(stub)
vision/detector.py            YOLOv8n vehicle detection + monocular distance
vision/frames.py              time-addressable frame sequences (clips)
vision/synthetic_clip.py      synthesized approach clip (real inference, not real footage)
fusion/aeb_pipeline.py        OBD speed + vision gap -> TTC -> AEB score
run_full_validation_suite.py  runs both tracks + the convergence check
cpp/                          C++17 control-loop core + pybind11 bindings + GoogleTest
data/                         the OBD-II dataset (gitignored)
reports/                      generated reports and plots (gitignored)
tests/                        143 Python tests
```

---

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.9+. First use of the vision module downloads the YOLOv8n
weights (~6 MB) into `vision/weights/` (gitignored).

```bash
# full Python test suite (143 tests)
python3 -m pytest -q

# build the C++ core, then run GoogleTest + the parity/equivalence tests
./cpp/build.sh

# Track 1 only — 29 simulated scenarios, one consolidated report
python3 -m scenarios.track1_suite

# Track 2 only — ingestion + vision + fusion backbone
python3 -m fusion.aeb_pipeline

# both tracks + the convergence check, with PDF output
python3 run_full_validation_suite.py          # add --no-pdf to skip PDFs
```

Reports and plots are written to `reports/` (gitignored as generated output).
Replay-mode tests skip cleanly if the OBD-II dataset isn't present, so the suite
passes on a fresh clone.

---

## Track 1 — Simulation

Three rule-based controllers driven through `highway-env` with a **directly
scripted** lead vehicle (steady / decel / accel / stop-and-go / oscillating /
cut-in), rather than the environment's random IDM traffic — a regression suite
whose scenarios drift is not a regression suite.

- **ACC** — proportional gap controller, scored on time-to-collision and
  *settled-state* following-distance error (mean over the final 20% of samples,
  since a transient right after a disturbance is expected, not a defect).
- **LKA** — PD lane-centering, tested on a straight segment (via `highway-v0`)
  and a winding road (a `SineLane` built directly, since `highway-v0` only makes
  straight roads).
- **AEB** — staged TTC braking that overrides ACC. One scenario replays the exact
  conditions of ACC's worst failure at matched control frequency and confirms
  AEB's wider authority avoids the collision ACC alone caused.

### Where the thresholds come from

The spec's original numbers were flagged as unsourced starting points and
replaced with published references before any scenario depended on them:

| Criterion | Value | Basis |
|---|---|---|
| AEB warning TTC | 2.0 s | NHTSA FCW criteria (decelerating / stopped lead) |
| AEB brake TTC | 1.8 s | NHTSA warning-abort threshold (~1.8–1.9 s) |
| Lane deviation | 0.3 m | **conservative proxy, not a direct match** — Euro NCAP scores via Distance-To-Lane-Edge, not offset from centre |

The schema docstring says exactly this. An earlier `max_brake_response_latency_s`
gate was *removed* when running the scenarios showed it was an unsourced number
that safe runs legitimately exceeded — it is still reported as a diagnostic.

---

## Track 2 — Real-vehicle software backbone

Nothing here touches hardware. It's the pipeline that *will*, built and exercised
against stand-ins chosen to be as real as possible.

- **`OBDReader`** — one interface, three modes. `mock` generates a synthetic
  drive cycle; `replay` plays back a real public dataset; `live` is a deliberate
  `NotImplementedError`. Swapping to `live` later is intended to be the only
  change needed.
- **Replay data** — a real 60,440-row public OBD-II dataset (14 drivers). It is
  genuinely messy: ~80% of rows missing RPM/throttle, ~280 fragmented trips per
  vehicle, irregular sampling, and **no brake signal at all**. The reader picks
  the best-covered trip, forward-fills small gaps, and derives a clearly-labelled
  brake heuristic. Handling that mess is the point — a pipeline that only ever
  saw the mock generator would break on contact with a real car.
- **Vision** — YOLOv8n filtered to vehicle classes, with a monocular distance
  estimate `(1.8 m × focal_length) / bbox_width_px`.
- **Fusion** — ego speed from OBD-II, gap distance from vision, combined into a
  TTC scored against the **same** AEB criteria as Track 1. Lead speed (which
  vision can't observe directly) is derived from the closing rate between
  consecutive gap readings, stamped with each reading's own measurement time so
  the rate is correct even when the camera updates slower than the control loop.

### On the synthesized vision clip

With no dashcam footage available, the end-to-end vision run uses a synthesized
*approach clip*: a real photograph composited at increasing scale so the vehicle
grows in frame. **YOLO runs on every frame for real** — the bounding boxes,
distances, and every downstream number are genuine detector output, not
hand-written. This proves the pipeline is **wired** correctly end to end. It is
**not** real footage and supports **no** accuracy claim; `vision/synthetic_clip.py`
says so explicitly. Swap in a `VideoFrameSequence` over a real clip and nothing
above it changes.

---

## Honest limitations

- **No run is labelled `real_world` yet** — correctly. A run earns that label
  only when *both* inputs are real (a live OBD-II drive paired with footage from
  that same drive). Replayed OBD-II with a synthesized clip is still labelled
  `simulation`. The reporting path for `real_world` is built and tested so it's
  ready the day hardware arrives.
- **The controllers are baselines, not products.** They are meant to be simple
  enough to defend, and the harness exists precisely to catch where they fall
  short.
- **The synthesized clip validates wiring, not detection accuracy** (see above).

## Deliberately out of scope

Per the project spec: no live Bluetooth/camera/drive-logging code (the hardware
phase); no merged cross-track report (the "Option C" step — the two tracks are
proven format-identical but not combined); and no Blind Spot Monitoring or Park
Assist, which `highway-env` models poorly. CARLA was rejected for lack of macOS
support.

## When the OBD-II adapter arrives

Swap `OBDReader(mode="mock")` for `OBDReader(mode="live")`, point a
`VideoFrameSequence` at real forward footage, and run the existing
`fusion/aeb_pipeline.py`. No pipeline code should need to change — only the data
sources. That claim is the whole design bet; the injection points exist and are
tested, but the bet itself stays untested until there's hardware to test it on.
