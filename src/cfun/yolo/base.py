import ast
from pathlib import Path
from typing import Union

import numpy as np
import onnx
from PIL import Image


def extract_info_from_onnx(onnx_path, key: str) -> Union[str, list, dict]:
    """从 ONNX 模型中提取信息

    Args:
        onnx_path (str): ONNX 模型路径
        key (str): 要提取的信息的键

    Returns:
        Union[str, list, dict]: 提取的信息,可以是字符串、列表或字典
    """
    model = onnx.load(onnx_path)
    metadata = {p.key: p.value for p in model.metadata_props}
    if key in metadata:
        return ast.literal_eval(metadata[key])  # 将原始的字符串转换为python对象
    else:
        raise ValueError(f"Key '{key}' not found in ONNX metadata.")


def load_image(source: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    """加载图像,支持路径、PIL.Image 或 OpenCV 图像（ndarray）,

    希望支持各种图片格式,最终转为 RGB 格式的 PIL.Image 对象, 只对单个图片进行处理

    Args:
        source (Union[str, Path, Image.Image, np.ndarray]): 图像路径或图像对象

    Returns:
        Image.Image: 加载后的图像对象

    """
    if isinstance(source, (str, Path)):
        img0 = Image.open(source).convert("RGB")
    elif isinstance(source, Image.Image):
        img0 = source.convert("RGB")
    elif isinstance(source, np.ndarray):
        import cv2

        if source.ndim == 3 and source.shape[2] == 3:
            img0 = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError("Unsupported ndarray format: must be HxWx3 BGR image")
    else:
        raise TypeError("source must be a file path, PIL.Image, or cv2 image (ndarray)")

    return img0


def fillter_top5(result: dict, threshold: float = 0.2) -> dict:
    # 根据 top5conf 过滤
    filtered = [
        (i, name, conf)
        for i, name, conf in zip(
            result["top5"], result["top5name"], result["top5conf"], strict=True
        )
        if conf > threshold
    ]

    # 如果你只要 top5 中符合条件的 id、name、conf 三个列表
    filtered_ids = [i for i, _, _ in filtered]
    filtered_names = [name for _, name, _ in filtered]
    filtered_confs = [conf for _, _, conf in filtered]
    # 如果你要返回一个字典
    filtered_dict = {
        "top5": filtered_ids,
        "top5name": filtered_names,
        "top5conf": filtered_confs,
    }
    return filtered_dict


def jaccard_similarity(
    set1: Union[set, list, tuple, str],
    set2: Union[set, list, tuple, str],
    precision: int = 4,
) -> float:
    """计算两个集合、列表、元组或字符串之间的 Jaccard 相似度。

    Jaccard 相似度 = 交集大小 / 并集大小

    Args:
        set1 (Union[set, list, tuple, str]): 第一个集合或可迭代对象
        set2 (Union[set, list, tuple, str]): 第二个集合或可迭代对象
        precision (int): 返回的小数点精度,默认保留4位

    Returns:
        float: Jaccard 相似度,范围 [0.0, 1.0]
    """
    try:
        set1_converted, set2_converted = set(set1), set(set2)
    except TypeError as e:
        raise ValueError(
            "输入必须是可转换为 set 的可迭代对象,如 list、tuple、str、set"
        ) from e

    union = set1_converted | set2_converted
    if not union:
        return 0.0  # 避免除以零

    intersection = set1_converted & set2_converted
    similarity = len(intersection) / len(union)
    return round(similarity, precision)
