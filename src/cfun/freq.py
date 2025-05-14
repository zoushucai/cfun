"""频率数据

加载中文词频数据

Attributes:
    FREQUENCY (pd.DataFrame): 包含频率数据的 DataFrame. 该 DataFrame 包含两列:

        - `word` 列: 单词（词语）
        - `count` 列: 单词（词语）的频率计数


Example:
    ```python
    from cfun.freq import FREQUENCY
    print(FREQUENCY.head())
    '''
        word     count
        0  有限公司  19486265
        1  固定资产   2347021
        2  无形资产   1942123
        3  股东大会   1712131
        4  上市公司   1645526
    '''
    ```
"""

import importlib.resources as pkg_resources

import pandas as pd

from . import data


def _load_parquet(filename: str) -> pd.DataFrame:
    """加载 parquet 文件

    该函数用于加载 parquet 文件并返回一个 pandas DataFrame

    Args:
        filename (str): 要加载的 parquet 文件名

    Returns:
        pd.DataFrame: 返回的 DataFrame
    """
    path = pkg_resources.files(data).joinpath(filename)
    with path.open("rb") as f:
        return pd.read_parquet(f)


FREQUENCY = _load_parquet("frequency.parquet")

__all__ = [
    "FREQUENCY",
]
