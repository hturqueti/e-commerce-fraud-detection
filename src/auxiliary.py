# Libraries
import pandas as pd
import yaml

from pathlib import Path
from typing import Optional

# Constants
project_path = Path('.').resolve().parent

def load_parameters() -> dict[str, str]:
    with open(project_path.joinpath("parameters.yaml"), 'r') as file:
        parameters = yaml.safe_load(file)
    return parameters

def filter_quantiles(
    df: pd.DataFrame,
    feature_filter: str,
    min_quantile: Optional[float] = None,
    max_quantile: Optional[float] = None
) -> pd.DataFrame:
    # Get min value
    """
    Filter the rows of a dataframe between two values of a column.

    The two values must be given as quantiles of the column.
    If the value is not given, it is set to the minimum or maximum of the column.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be filtered.
    feature_filter : str
        The column to be used for filtering.
    min_quantile : Optional[float], optional
        The quantile of the minimum value, by default None.
    max_quantile : Optional[float], optional
        The quantile of the maximum value, by default None.

    Returns
    -------
    pd.DataFrame
        The filtered dataframe.
    """
    if min_quantile is None:
        min_x = df[feature_filter].min()
    else:
        min_x = df[feature_filter].quantile(min_quantile)

    # Get max value
    if max_quantile is None:
        max_x = df[feature_filter].max()
    else:
        max_x = df[feature_filter].quantile(max_quantile)

    return df.loc[df[feature_filter].between(min_x, max_x), :]