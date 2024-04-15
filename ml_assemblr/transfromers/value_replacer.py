from ml_assemblr.main_components.data_pod import DataPod
from ml_assemblr.main_components.transformer import DataFrameTransformer


class ValueReplacer(DataFrameTransformer):
    """A transformer for replacing values in specified columns of a DataFrame.

    This transformer replaces specified values in specified columns of a DataFrame
    with new values according to provided replacing mappers.

    Parameters
    ----------
    replacing_mappers : dict[str, dict]
        A dictionary where keys are column names and values are dictionaries representing
        the mapping of values to be replaced with their corresponding replacements.
    """

    replacing_mappers: dict[str, dict]

    def _fit_transform(self, data_pod: DataPod) -> DataPod:
        df = data_pod.dfs[self.target_df_name]
        for column, replacing_mapper in self.replacing_mappers.items():
            df[column] = df[column].replace(replacing_mapper)

        assert isinstance(self.target_df_name, str)
        data_pod.df_nodes[self.target_df_name].df = df
        return data_pod

    def _transform(self, data_pod: DataPod) -> DataPod:
        return self._fit_transform(data_pod)
