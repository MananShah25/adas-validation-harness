from reporting.defect_log import Defect, DefectLog, Severity
from reporting.report_builder import ReportBuilder
from schemas.test_procedure import DataSource, Feature, TestProcedure, TestResult


def test_end_to_end_report(tmp_path):
    procedure = TestProcedure(
        feature=Feature.AEB,
        scenario_id="aeb_lead_stopped_01",
        description="Lead vehicle stops suddenly at 50 km/h",
        preconditions={"ego_speed_kmh": 50},
        pass_criteria={"warning_time_to_collision_s": 2.0},
        data_source=DataSource.SIMULATION,
    )
    result = TestResult(
        procedure=procedure,
        passed=False,
        observed_metrics={"min_ttc_s": 1.2},
        time_series={"ttc": [3.0, 2.5, 1.8, 1.2]},
        sample_times=[0.0, 0.5, 1.0, 1.5],
    )

    defect_log = DefectLog(tmp_path / "defects.csv")
    defect_log.log(
        Defect(
            feature=Feature.AEB,
            scenario_id="aeb_lead_stopped_01",
            expected="TTC stays above 2.0s warning threshold",
            observed="TTC dropped to 1.2s before warning fired",
            severity=Severity.HIGH,
            reproduction_steps="Run aeb_lead_stopped_01 scenario",
            data_source=DataSource.SIMULATION,
        )
    )

    builder = ReportBuilder(output_dir=tmp_path / "reports")
    report_path = builder.build([result], defect_log=defect_log)

    assert report_path.exists()
    content = report_path.read_text()
    assert "AEB" in content
    assert "FAIL" in content
    assert "DEF-0001" in content
    assert (tmp_path / "reports" / "plots").exists()
    assert list((tmp_path / "reports" / "plots").glob("*.png"))


def test_report_without_defect_log_or_plots(tmp_path):
    procedure = TestProcedure(
        feature=Feature.ACC,
        scenario_id="acc_steady_01",
        description="Steady-state car following",
        preconditions={},
        pass_criteria={"min_time_to_collision_s": 2.0},
        data_source=DataSource.SIMULATION,
    )
    result = TestResult(procedure=procedure, passed=True, observed_metrics={"min_ttc_s": 4.5})

    builder = ReportBuilder(output_dir=tmp_path / "reports")
    report_path = builder.build([result])

    content = report_path.read_text()
    assert "PASS" in content
    assert "ACC" in content
