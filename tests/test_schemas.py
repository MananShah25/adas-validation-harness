from schemas.test_procedure import DataSource, Feature, TestProcedure


def test_feature_and_data_source_accept_strings():
    proc = TestProcedure(
        feature="ACC",
        scenario_id="acc_01",
        description="test",
        preconditions={},
        pass_criteria={},
        data_source="simulation",
    )
    assert proc.feature is Feature.ACC
    assert proc.data_source is DataSource.SIMULATION


def test_feature_and_data_source_accept_enums():
    proc = TestProcedure(
        feature=Feature.AEB,
        scenario_id="aeb_01",
        description="test",
        preconditions={},
        pass_criteria={},
        data_source=DataSource.REAL_WORLD,
    )
    assert proc.feature is Feature.AEB
    assert proc.data_source is DataSource.REAL_WORLD
