"""Batch report generator.

Takes a batch of TestResults plus an optional DefectLog and produces a
single Markdown summary: pass/fail counts per feature, a scenario table,
basic time-series plots, and the defect list.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reporting.defect_log import DefectLog
from schemas.test_procedure import TestResult


class ReportBuilder:
    def __init__(self, output_dir: str | Path = "reports"):
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def _plot_time_series(self, result: TestResult) -> list[Path]:
        paths: list[Path] = []
        if not result.time_series or not result.sample_times:
            return paths
        for metric_name, values in result.time_series.items():
            if len(values) != len(result.sample_times):
                continue
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(result.sample_times, values)
            ax.set_xlabel("time (s)")
            ax.set_ylabel(metric_name)
            ax.set_title(f"{result.procedure.scenario_id} — {metric_name}")
            fig.tight_layout()
            plot_path = self.plots_dir / f"{result.procedure.scenario_id}_{metric_name}.png"
            fig.savefig(plot_path)
            plt.close(fig)
            paths.append(plot_path)
        return paths

    def build(
        self,
        results: list[TestResult],
        defect_log: DefectLog | None = None,
    ) -> Path:
        counts: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "failed": 0})
        for r in results:
            key = r.procedure.feature.value
            counts[key]["passed" if r.passed else "failed"] += 1

        lines = [
            "# ADAS Validation Report",
            "",
            f"Generated: {datetime.utcnow().isoformat()}Z",
            "",
            "## Summary",
            "",
            "| Feature | Passed | Failed | Total | Pass rate |",
            "|---|---|---|---|---|",
        ]
        for feature, c in sorted(counts.items()):
            total = c["passed"] + c["failed"]
            rate = f"{(c['passed'] / total * 100):.0f}%" if total else "n/a"
            lines.append(f"| {feature} | {c['passed']} | {c['failed']} | {total} | {rate} |")
        lines.append("")

        lines += [
            "## Scenario results",
            "",
            "| Scenario | Feature | Data source | Result | Observed metrics |",
            "|---|---|---|---|---|",
        ]
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in r.observed_metrics.items())
            lines.append(
                f"| {r.procedure.scenario_id} | {r.procedure.feature.value} | "
                f"{r.procedure.data_source.value} | {status} | {metrics_str} |"
            )
        lines.append("")

        plot_paths: list[Path] = []
        for r in results:
            plot_paths += self._plot_time_series(r)
        if plot_paths:
            lines.append("## Plots")
            lines.append("")
            for p in plot_paths:
                rel = p.relative_to(self.output_dir)
                lines.append(f"![{p.stem}]({rel})")
            lines.append("")

        if defect_log is not None:
            lines.append("## Defects")
            lines.append("")
            lines.append(defect_log.to_markdown())

        report_path = self.output_dir / f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
        report_path.write_text("\n".join(lines))
        return report_path
