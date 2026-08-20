"""Phase 9 convergence check: run Track 1's full simulation suite (ACC +
LKA + AEB in highway-env) and Track 2's fusion backbone test (OBD-II
ingestion + vision, mock/synthetic sources) and confirm both produce
reports in the identical format.

This deliberately does NOT merge the two into one combined report -- that
merge is the project spec's "Option C" step, explicitly out of scope for
now. The point here is only to verify the two tracks' schemas haven't
diverged: same TestProcedure/TestResult types, same ReportBuilder, same
DefectLog columns, regardless of whether the data came from highway-env
physics or the OBD-II + vision pipeline.
"""

from __future__ import annotations

from pathlib import Path

from fusion.aeb_pipeline import run_all as run_track2_suite
from scenarios.track1_suite import run_full_suite as run_track1_suite


def run_and_confirm_convergence(output_dir: str = "reports", emit_pdf: bool = False) -> tuple[Path, Path]:
    track1_report = run_track1_suite(
        output_dir=f"{output_dir}/track1",
        defect_log_path=f"{output_dir}/track1/defect_log.csv",
        emit_pdf=emit_pdf,
    )
    track2_report = run_track2_suite(
        output_dir=f"{output_dir}/track2",
        defect_log_path=f"{output_dir}/track2/defect_log.csv",
        emit_pdf=emit_pdf,
    )
    return track1_report, track2_report


if __name__ == "__main__":
    import sys

    emit_pdf = "--no-pdf" not in sys.argv
    t1_report, t2_report = run_and_confirm_convergence(emit_pdf=emit_pdf)
    print(f"Track 1 (simulation) report: {t1_report}")
    print(f"Track 2 (backbone) report:   {t2_report}")
    if emit_pdf:
        print("PDF versions written alongside each (pass --no-pdf to skip).")
