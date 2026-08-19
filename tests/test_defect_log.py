from reporting.defect_log import Defect, DefectLog, Severity
from schemas.test_procedure import DataSource, Feature


def _sample_defect(scenario_id: str) -> Defect:
    return Defect(
        feature=Feature.ACC,
        scenario_id=scenario_id,
        expected="e",
        observed="o",
        severity=Severity.LOW,
        reproduction_steps="steps",
        data_source=DataSource.SIMULATION,
    )


def test_defect_id_auto_increments_and_persists_across_reopen(tmp_path):
    log_path = tmp_path / "defects.csv"
    log = DefectLog(log_path)
    d1 = log.log(_sample_defect("s1"))
    d2 = log.log(_sample_defect("s2"))
    assert d1.defect_id == "DEF-0001"
    assert d2.defect_id == "DEF-0002"

    log2 = DefectLog(log_path)
    d3 = log2.log(_sample_defect("s3"))
    assert d3.defect_id == "DEF-0003"


def test_to_markdown_renders_expected_observed(tmp_path):
    log = DefectLog(tmp_path / "defects.csv")
    log.log(_sample_defect("s1"))
    md = log.to_markdown()
    assert "DEF-0001" in md
    assert "**Expected:** e" in md
    assert "**Observed:** o" in md


def test_to_markdown_empty_log(tmp_path):
    log = DefectLog(tmp_path / "defects.csv")
    assert "No defects logged" in log.to_markdown()
