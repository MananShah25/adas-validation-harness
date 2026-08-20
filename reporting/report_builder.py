"""Batch report generator.

Takes a batch of TestResults plus an optional DefectLog and produces a
single summary -- pass/fail counts per feature, a scenario table, basic
time-series plots, and the defect list -- as Markdown (build) or as a
paginated PDF (build_pdf).

The PDF path goes through matplotlib's PdfPages rather than adding an HTML
or LaTeX toolchain, since matplotlib is already a dependency for the plots
and this keeps the report generator runnable anywhere the suite runs.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from reporting.defect_log import DefectLog
from schemas.test_procedure import TestResult

A4_INCHES = (8.27, 11.69)
LINES_PER_PAGE = 58


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

    @staticmethod
    def _feature_counts(results: list[TestResult]) -> dict[str, dict[str, int]]:
        """Pass/fail tally per feature. Shared by every output format so the
        Markdown and PDF reports cannot disagree about the same run."""
        counts: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "failed": 0})
        for r in results:
            counts[r.procedure.feature.value]["passed" if r.passed else "failed"] += 1
        return counts

    @staticmethod
    def _scenario_row(result: TestResult) -> tuple[str, str, str, str, str]:
        """One scenario's cells, in the order every format renders them."""
        metrics = ", ".join(f"{k}={v:.3f}" for k, v in result.observed_metrics.items())
        return (
            result.procedure.scenario_id,
            result.procedure.feature.value,
            result.procedure.data_source.value,
            "PASS" if result.passed else "FAIL",
            metrics,
        )

    def build(
        self,
        results: list[TestResult],
        defect_log: DefectLog | None = None,
    ) -> Path:
        counts = self._feature_counts(results)

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
            scenario, feature, source, status, metrics_str = self._scenario_row(r)
            lines.append(f"| {scenario} | {feature} | {source} | {status} | {metrics_str} |")
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

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    @staticmethod
    def _text_pages(lines: list[str], pdf: PdfPages) -> None:
        """Write `lines` across as many monospaced A4 pages as needed."""
        for start in range(0, len(lines), LINES_PER_PAGE):
            chunk = lines[start:start + LINES_PER_PAGE]
            fig = plt.figure(figsize=A4_INCHES)
            y = 0.96
            for line in chunk:
                fig.text(0.06, y, line, family="monospace", fontsize=7.5, va="top")
                y -= 0.0163
            pdf.savefig(fig)
            plt.close(fig)

    def _pdf_text_lines(self, results: list[TestResult], defect_log: DefectLog | None) -> list[str]:
        counts = self._feature_counts(results)

        lines = [
            "ADAS VALIDATION REPORT",
            f"Generated: {datetime.utcnow().isoformat()}Z",
            "",
            "SUMMARY",
            "-" * 92,
            f"{'Feature':<10}{'Passed':>8}{'Failed':>8}{'Total':>8}{'Pass rate':>12}",
        ]
        for feature, c in sorted(counts.items()):
            total = c["passed"] + c["failed"]
            rate = f"{(c['passed'] / total * 100):.0f}%" if total else "n/a"
            lines.append(f"{feature:<10}{c['passed']:>8}{c['failed']:>8}{total:>8}{rate:>12}")

        lines += ["", "SCENARIO RESULTS", "-" * 92]
        for r in results:
            scenario, feature, source, status, metrics = self._scenario_row(r)
            lines.append(f"[{status}] {scenario}  ({feature} / {source})")
            # Metrics wrap rather than run off the page edge.
            for i in range(0, len(metrics), 84):
                lines.append(f"        {metrics[i:i + 84]}")

        if defect_log is not None:
            defects = defect_log.read_all()
            lines += ["", "DEFECTS", "-" * 92]
            if not defects:
                lines.append("No defects logged.")
            for d in defects:
                lines.append(
                    f"{d['defect_id']}  {d['feature']} / {d['scenario_id']}  [{d['severity'].upper()}]"
                )
                lines.append(f"        source   : {d['data_source']}")
                lines.append(f"        steps    : {d['reproduction_steps'][:80]}")
                lines.append(f"        expected : {d['expected'][:80]}")
                lines.append(f"        observed : {d['observed'][:80]}")
                lines.append("")

        return lines

    def build_pdf(
        self,
        results: list[TestResult],
        defect_log: DefectLog | None = None,
    ) -> Path:
        """The same run as build(), rendered as a paginated PDF with the
        time-series plots embedded as full pages rather than linked files."""
        pdf_path = self.output_dir / f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"

        with PdfPages(pdf_path) as pdf:
            self._text_pages(self._pdf_text_lines(results, defect_log), pdf)

            for result in results:
                if not result.time_series or not result.sample_times:
                    continue
                plottable = {
                    name: values
                    for name, values in result.time_series.items()
                    if len(values) == len(result.sample_times)
                }
                if not plottable:
                    continue

                fig, axes = plt.subplots(
                    len(plottable), 1, figsize=A4_INCHES, sharex=True, squeeze=False
                )
                for ax, (name, values) in zip(axes[:, 0], plottable.items()):
                    ax.plot(result.sample_times, values, linewidth=1.2)
                    ax.set_ylabel(name, fontsize=8)
                    ax.tick_params(labelsize=7)
                    ax.grid(alpha=0.3)
                axes[-1, 0].set_xlabel("time (s)", fontsize=8)
                axes[0, 0].set_title(
                    f"{result.procedure.scenario_id} ({'PASS' if result.passed else 'FAIL'})",
                    fontsize=10,
                )
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        return pdf_path
