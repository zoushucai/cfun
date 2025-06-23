"""与加密相关的函数

该模块提供了计算文件 MD5 值的函数。

Functions:
    calculate_md5: 计算文件的 MD5 值

"""

import hashlib
from pathlib import Path


def calculate_md5(file_path: Path | str) -> str:
    """计算文件的 MD5 值

    Args:
        file_path (Path | str): 文件路径,可以是 Path 对象或字符串

    Returns:
        str: 文件的 MD5 值

    Raises:
        AssertionError: 文件不存在或路径不是文件或文件大小为0

    Example:
        ```python

        from cfun.md5 import calculate_md5
        file_path = "example.txt"
        md5_value = calculate_md5(file_path)
        print(md5_value) # 5d41402abc4b2a76b9719d911017c592

        ```
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    assert file_path.exists(), f"文件不存在: {file_path}"
    assert file_path.is_file(), f"路径不是文件: {file_path}"
    assert file_path.stat().st_size > 0, f"文件大小为0: {file_path}"
    hash_md5 = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
