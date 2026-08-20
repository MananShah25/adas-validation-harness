"""Full Track 1 regression suite: runs ACC, LKA, and AEB scenarios together
and produces one consolidated report and defect log, confirming all three
features go through the exact same shared schema/reporting pipeline.
"""

from __future__ import annotations

from pathlib import Path

from reporting.defect_log import Defect, DefectLog, Severity
from reporting.report_builder import ReportBuilder
from schemas.test_procedure import DataSource, Feature, TestResult
from scenarios.acc_scenarios import build_acc_scenarios, run_acc_scenario
from scenarios.aeb_scenarios import build_aeb_scenarios, run_aeb_scenario
from scenarios.lka_scenarios import build_lka_scenarios, run_lka_scenario


def _severity_for(result: TestResult) -> Severity:
    feature = result.procedure.feature
    metrics = result.observed_metrics
    if feature is Feature.AEB:
        return Severity.CRITICAL if metrics.get("collided", 0.0) else Severity.HIGH
    if feature is Feature.ACC:
        return Severity.HIGH if metrics.get("min_ttc_s", 99.0) < 1.0 else Severity.MEDIUM
    if feature is Feature.LKA:
        threshold = result.procedure.pass_criteria.get("max_lane_deviation_m", 0.3)
        return Severity.HIGH if metrics.get("max_lane_deviation_m", 0.0) > 2 * threshold else Severity.MEDIUM
    return Severity.MEDIUM


def _log_defect(defect_log: DefectLog, result: TestResult) -> None:
    feature = result.procedure.feature
    defect_log.log(
        Defect(
            feature=feature,
            scenario_id=result.procedure.scenario_id,
            expected=f"pass_criteria satisfied: {result.procedure.pass_criteria}",
            observed=f"observed_metrics: {result.observed_metrics}",
            severity=_severity_for(result),
            reproduction_steps=(
                f"run_{feature.value.lower()}_scenario(spec) for scenario_id="
                f"'{result.procedure.scenario_id}' in scenarios/{feature.value.lower()}_scenarios.py"
            ),
            data_source=DataSource.SIMULATION,
        )
    )


def run_full_suite(output_dir: str = "reports", defect_log_path: str = "reports/defect_log.csv") -> Path:
    defect_log = DefectLog(defect_log_path)

    results: list[TestResult] = []
    results += [run_acc_scenario(spec) for spec in build_acc_scenarios()]
    results += [run_lka_scenario(spec) for spec in build_lka_scenarios()]
    results += [run_aeb_scenario(spec) for spec in build_aeb_scenarios()]

    for result in results:
        if not result.passed:
            _log_defect(defect_log, result)

    builder = ReportBuilder(output_dir=output_dir)
    return builder.build(results, defect_log=defect_log)


if __name__ == "__main__":
    report_path = run_full_suite()
    print(f"Consolidated Track 1 report written to {report_path}")
