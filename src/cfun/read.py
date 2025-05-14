"""读取文件的函数

该模块提供了读取csv和txt文件的函数，并支持并行处理。

"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

## 对于3.12的版本， 可以使用快捷的批处理函数：
# from itertools import batched


######################################################
########## 公共部分 ##################
###########################################
## 这里定义一个 batched函数， 用于将一个迭代器分成多个批次， python3.12版本可以直接使用: from itertools import batched
def batched(iterable: Iterable, n: int):
    """Returns a batch of n items at a time.

    如果在python3.12版本中， 可以直接使用: from itertools import batched

    Args:
        iterable: An iterable object.
        n: The size of each batch.
    Yields:
        A batch of n items from the iterable.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def parallel_handle(
    Iterative: Iterable,
    func: Callable,
    args: tuple = (),
    max_workers: Optional[int] = None,
) -> list:
    """并行处理函数的一个简单的模板

    Args:
        Iterative (Iterable): 要并行处理的迭代对象。
        func (Callable): 要执行的函数，签名应为 func(item, *args)。
        args (tuple): 传递给函数的额外参数。
        max_workers (int, optional): 并发进程数，默认由系统决定。

    Returns:
        处理结果列表

    Example:
        ```python
        #多进程需要在主线程中运行
        from cfun.read import parallel_handle
        if __name__ == "__main__":
            def process_item(item, factor, add_value):
                return item * factor + add_value
            items = [1, 2, 3, 4]
            factor = 10
            add_value = 5
            args = (factor, add_value)
            # 传递 items, process_item 函数和 args 元组
            results = parallel_handle(items, process_item, args=args)
            print(results)  # 输出 [15, 25, 35, 45]
        ```
    """

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, item, *args): item for item in Iterative}

        for future in as_completed(futures):
            try:
                item = futures[future]
                results.append(future.result())
            except Exception as e:
                raise AssertionError(f"处理失败，输入项: {item}, 错误信息: {e}") from e

    return results


######################################################
########## csv ##################
###########################################


def _load_csv(file: str | Path, encoding: str = "utf-8") -> pd.DataFrame:
    """读取csv文件
    Args:
        file: 文件路径
        encoding: 编码格式

    Returns:
        DataFrame: 读取的DataFrame
    """
    return pd.read_csv(file, encoding=encoding)


def parallel_load_csv(
    files: list[str] | list[Path],
    encoding: str = "utf-8",
    max_workers: Optional[int] = None,
    batch_size: int = 6,
) -> pd.DataFrame:
    """并行处理csv文件

    Args:
        files (list[str] | list[Path]): 文件路径列表
        encoding: 编码格式
        max_workers (int): 最大工作线程数
        batch_size (int): 每批处理的文件数， 当设置很大的时候，即全部文件一起处理时，（前提每个文件都很小），

    Returns:
        DataFrame: 读取的DataFrame
    """
    all_dfs = []
    batched_files = batched(files, batch_size)
    for _idx, batch in enumerate(batched_files):
        dfs = parallel_handle(
            batch, _load_csv, args=(encoding,), max_workers=max_workers
        )
        if dfs:
            all_dfs.append(pd.concat(dfs, ignore_index=True))
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


######################################################
########## txt ##################
###########################################
def _load_txt(file_path: str, encoding: str = "utf-8") -> str:
    """加载文本文件，返回字符串内容。

    Args:
        file_path (str): 文件路径
        encoding (str): 文件编码，默认为utf-8

    Returns:
        str: 文件内容,字符串
    """
    assert Path(file_path).exists(), f"文件不存在：{file_path}"
    with open(file_path, "r", encoding=encoding, errors="ignore") as f:
        return f.read()


def parallel_load_txt(
    files: list[str] | list[Path],
    encoding="utf-8",
    max_workers: Optional[int] = None,
    batch_size: int = 6,
) -> str:
    """并行处理txt文件

    Args:
        files (list[str] | list[Path]): 文件路径列表
        encoding (str): 编码格式
        max_workers (int): 最大工作线程数
        batch_size (int): 每批处理的文件数， 当设置很大的时候，即全部文件一起处理时，（前提每个文件都很小），

    Returns:
        str: 读取的文件内容，字符串
    """
    total = len(files)
    all_txts = []
    for _idx, start in enumerate(range(0, total, batch_size)):
        end = min(start + batch_size, total)
        batch = files[start:end]
        txts = parallel_handle(
            batch, _load_txt, args=(encoding,), max_workers=max_workers
        )
        if txts:
            all_txts.append("".join(txts))

    return "".join(all_txts) if all_txts else ""


if __name__ == "__main__":

    def process_item(item, factor, add_value):
        return item * factor + add_value

    items = [1, 2, 3, 4]
    factor = 10
    add_value = 5
    args = (factor, add_value)

    # 传递 items, process_item 函数和 args 元组
    results = parallel_handle(items, process_item, args=args)
    print(results)  # 输出 [15, 25, 35, 45]
