from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.main_components.transformer import Serializer
from ml_assemblr.transformer.unfit.bicolumns_operator import BicolumnsOperator


def test_data_serializer(some_dp: DataPod):
    prod_dp = some_dp.copy()
    transformer_1 = BicolumnsOperator(
        target_col_name="output_1", first_col_name="age", second_col_name="salary", operation="+"
    )
    transformer_2 = BicolumnsOperator(
        target_col_name="output_2", first_col_name="age", second_col_name="salary", operation="+"
    )

    pipeline = Serializer(transformers=[transformer_1, transformer_2])

    assert "output_1" not in some_dp.main_df.columns
    assert "output_2" not in some_dp.main_df.columns
    some_dp = some_dp.fit_transform(pipeline)
    assert "output_1" in some_dp.main_df.columns
    assert "output_2" in some_dp.main_df.columns

    assert "output_1" not in prod_dp.main_df.columns
    assert "output_2" not in prod_dp.main_df.columns
    prod_dp = prod_dp.transform(some_dp.footprints)
    assert "output_1" in prod_dp.main_df.columns
    assert "output_2" in prod_dp.main_df.columns


def test_data_serializer_flatting(some_dp: DataPod):
    transformers = []

    for i in range(4):
        transformers.append(
            BicolumnsOperator(
                target_col_name=f"output_{i}",
                first_col_name="age",
                second_col_name="salary",
                operation="+",
            )
        )

    pipeline_1 = Serializer(transformers=[transformers[2], transformers[3]])
    pipeline_2 = Serializer(transformers=[transformers[0], transformers[1], pipeline_1])

    assert len(pipeline_2.transformers) == 4

    for transformer in pipeline_2.transformers:
        assert isinstance(transformer, BicolumnsOperator)

    for i in range(4):
        assert f"output_{i}" not in some_dp.main_df.columns

    some_dp = some_dp.fit_transform(pipeline_2)

    for i in range(4):
        assert f"output_{i}" in some_dp.main_df.columns


def test_data_frame_transformer_on_support_df(some_dp: DataPod):
    prod_dp = some_dp.copy()
    transformer_for_support_df = BicolumnsOperator(
        target_df_name="some_support_df",
        target_col_name="output",
        first_col_name="hours_worked",
        second_col_name="hours_worked",
        operation="+",
    )
    assert "output" not in some_dp.dfs["some_support_df"].columns
    some_dp = some_dp.fit_transform(transformer_for_support_df)
    assert "output" in some_dp.dfs["some_support_df"].columns

    assert "output" not in prod_dp.dfs["some_support_df"].columns
    prod_dp = prod_dp.transform(some_dp.footprints.transformers[-1])
    assert "output" in prod_dp.dfs["some_support_df"].columns
