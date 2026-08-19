"""Append-only defect log shared by both validation tracks."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from schemas.test_procedure import DataSource, Feature


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Defect:
    feature: Feature
    scenario_id: str
    expected: str
    observed: str
    severity: Severity
    reproduction_steps: str
    data_source: DataSource
    defect_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


_FIELDNAMES = [
    "defect_id",
    "feature",
    "scenario_id",
    "timestamp",
    "expected",
    "observed",
    "severity",
    "reproduction_steps",
    "data_source",
]


class DefectLog:
    """Append-only CSV defect log, reopenable across runs without resetting IDs."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=_FIELDNAMES).writeheader()

    def _next_defect_id(self) -> str:
        with self.path.open("r", newline="") as f:
            count = sum(1 for _ in csv.DictReader(f))
        return f"DEF-{count + 1:04d}"

    def log(self, defect: Defect) -> Defect:
        if not defect.defect_id:
            defect.defect_id = self._next_defect_id()
        row = asdict(defect)
        row["feature"] = defect.feature.value
        row["severity"] = defect.severity.value
        row["data_source"] = defect.data_source.value
        row["timestamp"] = defect.timestamp.isoformat()
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=_FIELDNAMES).writerow(row)
        return defect

    def read_all(self) -> list[dict]:
        with self.path.open("r", newline="") as f:
            return list(csv.DictReader(f))

    def to_markdown(self) -> str:
        defects = self.read_all()
        if not defects:
            return "_No defects logged._\n"
        lines = []
        for d in defects:
            lines.append(
                f"### {d['defect_id']} — {d['feature']} / {d['scenario_id']} "
                f"({d['severity'].upper()})"
            )
            lines.append("")
            lines.append(f"- **Data source:** {d['data_source']}")
            lines.append(f"- **Timestamp:** {d['timestamp']}")
            lines.append(f"- **Steps to reproduce:** {d['reproduction_steps']}")
            lines.append(f"- **Expected:** {d['expected']}")
            lines.append(f"- **Observed:** {d['observed']}")
            lines.append("")
        return "\n".join(lines)
