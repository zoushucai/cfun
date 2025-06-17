"""
通过json文件把 目标 生成yolo格式的txt文件, 进行目标检测数据分类,按指定的key进行分类, 比如：按照json文件中的 label字段进行分类

也可以直接用xlabeling 这些工具直接导出txt文件, 但需要手动划分数据集
"""

import json
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image


def json_to_yolo_txt(
    json_dir: Union[Path, str],
    label_key: str,
    class_mapping: dict[int, str],
    output_dir: Union[Path, str],
    image_dir: None | Path | str = None,
    image_suffix: str = ".png",
    force_overwrite: bool = False,
    ischeck: bool = True,
    shape_type: str = "rectangle",
) -> None:
    """将 XLabeling 标注的 JSON 文件转换为 YOLO 格式的 TXT 文件。

    Args:
        json_dir (Union[Path, str]): 存放 JSON 标注文件的目录
        label_key (str): 用于分类的键,如 "label"
        class_mapping (dict[int, str]): 类别映射字典, 键为整数 ID,从0开始,值为字符串类名,类别名称映射字典,对应json中的label_key
        output_dir (Union[Path, str]): 输出的 TXT 文件的目录
        image_dir (None | Path | str): 原始图像文件所在目录. json文件和图像文件同名
        image_suffix (str): 图像文件后缀名, 默认 ".png"
        force_overwrite (bool): 是否强制覆盖输出目录, 默认 False
        ischeck (bool): 是否检查图像文件是否存在, 默认 True, 如果为True, 则会根据json文件的名称去检查图像文件是否存在, 如果不存在则报错
        shape_type (str, optional): 形状类型. 暂时只能是rectangle(默认) 或 rotation.  rectangle表示矩形,rotation表示旋转矩形.

    Returns:
        None

    Example:
        ```python
        from cfun.yolo.convert import json_to_yolo_txt
        json_dir = "imgsdata/xlabeljson"
        label_key = "label"  # 指定分类的key
        class_mapping = {0: "1"}
        output_dir = "Label"  # 输出的txt文件的路径

        image_dir = "imgsdata"
        image_suffix = ".png"
        force_overwrite = True  # 是否强制覆盖输出目录
        json_to_yolo_txt(
            json_dir=json_dir,
            label_key=label_key,
            class_mapping=class_mapping,
            output_dir=output_dir,
            image_dir=image_dir,
            image_suffix=image_suffix,
            force_overwrite=force_overwrite,
            shape_type="rectangle",
        )
        ```
    """
    assert shape_type in {"rectangle", "rotation"}, (
        f"Unsupported shape_type: {shape_type}"
    )

    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    if ischeck and image_dir is not None:
        image_dir = Path(image_dir)
    clm = class_mapping  # 重新更改变量名,太长了
    if not isinstance(clm, dict) or not all(
        isinstance(k, int) and isinstance(v, str) for k, v in clm.items()
    ):
        raise ValueError(f"Invalid class_mapping format: {clm}")
    # 管理输出目录
    if output_dir.exists():
        if force_overwrite:
            shutil.rmtree(output_dir)
            print(f"[警告] 输出目录已存在,已被删除: {output_dir}")
        else:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取 JSON 文件和对应的图像文件
    json_files = list(json_dir.glob("*.json"))
    if ischeck and image_dir:
        image_dir = Path(image_dir)
        for json_file in json_files:
            image_path = image_dir / (json_file.stem + image_suffix)
            if not image_path.exists():
                raise FileNotFoundError(f"对应图像文件不存在: {image_path}")

    for idx, json_file in enumerate(json_files):
        if idx % 100 == 0 or idx == len(json_files) - 1:
            print(f"Processing {idx}/{len(json_files)}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        imageWidth, imageHeight = data["imageWidth"], data["imageHeight"]
        annotations = []

        for shape in data.get("shapes", []):
            points = shape["points"]
            label_name = shape[label_key]  # 类名
            label_id = next((k for k, v in clm.items() if v == label_name), None)
            assert label_id is not None, f"Label '{label_name}' not in class mapping."
            if shape_type == "rectangle":
                # YOLO assumes (x_center, y_center, width, height)
                # (x1, y1) 左上角,(x2, y2) 右下角
                x1, y1 = map(int, points[0])
                x2, y2 = map(int, points[2])
                assert x1 < x2 and y1 < y2, f"Invalid box: {x1, y1, x2, y2}"
                x_center = (x1 + x2) / 2 / imageWidth
                y_center = (y1 + y2) / 2 / imageHeight
                width = (x2 - x1) / imageWidth
                height = (y2 - y1) / imageHeight
                annotations.append((label_id, x_center, y_center, width, height))
            elif shape_type == "rotation":
                # 旋转矩形的处理
                norm_points = [(x / imageWidth, y / imageHeight) for x, y in points[:4]]
                # 展平列表
                annotations.append((label_id, *sum(norm_points, ())))
        txt_file = output_dir / (json_file.stem + ".txt")

        with open(txt_file, "w", encoding="utf-8") as f:
            for ann in annotations:
                f.write(f"{ann[0]} " + " ".join(f"{p:.6f}" for p in ann[1:]) + "\n")


def _crop_rotated_box(image: Image.Image, points: list) -> Image.Image:
    """
    裁剪旋转矩形图像区域。

    Args:
        image (PIL.Image): 原始图像
        points (list): 旋转框的四个点 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    Returns:
        PIL.Image: 裁剪后的图像
    """
    pts = np.array(points, dtype=np.float32)

    # 计算目标宽高
    width_top = float(np.linalg.norm(pts[0] - pts[1]))
    width_bottom = float(np.linalg.norm(pts[3] - pts[2]))
    height_left = float(np.linalg.norm(pts[0] - pts[3]))
    height_right = float(np.linalg.norm(pts[1] - pts[2]))
    width = int(max(width_top, width_bottom))
    height = int(max(height_left, height_right))

    # 目标坐标（拉直后）
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    # 仿射变换
    matrix = cv2.getPerspectiveTransform(pts, dst)
    img_cv = np.array(image)  # PIL -> np.array
    warped = cv2.warpPerspective(img_cv, matrix, (width, height))

    return Image.fromarray(warped)


def crop_images(
    json_dir: Union[Path, str],
    image_dir: Union[Path, str],
    image_suffix: str = ".png",
    category_key: str = "label",
    isremove_chinese: bool = True,
    output_dir: Union[Path, str] = "cropped",
    force_overwrite: bool = False,
    ischeck: bool = True,
    shape_type: str = "rectangle",
) -> None:
    """
    根据 JSON 标注文件裁剪图像,并根据指定字段（如 label ）分类保存。 (只支持矩形框的裁剪,旋转矩阵未知）

    Args:
        json_dir (Union[Path, str]): 存放 JSON 标注文件的目录
        image_dir (Union[Path, str]): 原始图像文件所在目录. json文件和图像文件同名,
        image_suffix (str): 图像文件后缀名, 默认 ".png"
        category_key (str): 用于分类图像的字段（例如 "label"）
        isremove_chinese (bool): 裁剪后的图片是否移除中文字符和下划线_, 默认 True, 这里的下划线是指原来文件名中的下划线,因为裁剪后的图片文件名中会有下划线
        output_dir (Union[Path, str]): 裁剪后图像的输出目录, 默认 "cropped"
        force_overwrite (bool): 是否强制覆盖输出目录, 默认 False, 如果为True, 则会删除原有的输出目录
        ischeck (bool): 是否检查图像文件是否存在, 默认 True, 如果为True, 检查json文件数量和图像文件数量是否一致, 如果不一致则报错

    Returns:
        None

    Example:
        ```python
        from cfun.yolo.convert import crop_images
        crop_images(
            json_dir="weilai1_rotation/weilai1_json_rotated",
            image_dir="weilai1_rotation",
            image_suffix=".jpg",
            category_key="label",
            isremove_chinese=True,
            output_dir="cropped_rotated",  # 输出的裁剪图片的路径
            force_overwrite=True,  # 是否强制覆盖输出目录
            ischeck = True,
            shape_type="rotation",
        )
        ```
    """

    json_dir = Path(json_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        if force_overwrite:
            shutil.rmtree(output_dir)
            print(f"[警告] 输出目录已存在,已被删除: {output_dir}")
        else:
            raise FileExistsError(f"Output directory already exists: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    # 查找所有 JSON 文件
    json_files = sorted(json_dir.glob("*.json"))
    if ischeck:
        for json_file in json_files:
            image_path = image_dir / (json_file.stem + image_suffix)
            if not image_path.exists():
                raise FileNotFoundError(f"缺失图像文件: {image_path}")
        print(f"[检查] 找到 {len(json_files)} 个 JSON 文件,图像文件匹配正常。")

    for idx, json_file in enumerate(json_files):
        if idx % 100 == 0 or idx == len(json_files) - 1:
            print(f"Processing {idx}/{len(json_files)}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_file = image_dir / (json_file.stem + image_suffix)
        try:
            image = Image.open(image_file)
        except Exception as e:
            print(f"[错误] 加载图像失败: {image_file}, 错误: {e}")
            continue
        base_name = json_file.stem
        if isremove_chinese:
            # 移除中文字符
            base_name = re.sub(r"[\u4e00-\u9fa5]", "", base_name)
            base_name = base_name.replace("_", "")  # 去掉下划线

        for shape in data.get("shapes", []):
            points = shape["points"]
            category = shape[category_key]
            if shape_type == "rectangle":
                x1, y1 = map(int, points[0])
                x2, y2 = map(int, points[2])
                assert x1 < x2 and y1 < y2, (
                    f"Invalid coordinates: {x1}, {y1}, {x2}, {y2}"
                )
                cropped = image.crop((x1, y1, x2, y2))
            elif shape_type == "rotation":
                cropped = _crop_rotated_box(image, points)
                x1, y1 = map(int, points[0])
            file_name = f"{base_name}_{x1}_{y1}{image_suffix}"
            out_path: Path = output_dir / category / file_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(out_path)


def _box_to_polygon(boxitem: dict, key: str = "box", replace: str = "points") -> dict:
    """将 box 表示 [x1, y1, x2, y2] 转换为 polygon 表示 [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]。 四个点来表示一个矩形框（左上,右上,右下,左下）.

    polygon对应四个点来表示一个矩形框（左上,右上,右下,左下）. box利用左上角和右下角的坐标来表示一个矩形框。 二者进行互换, 只是替换key以及对应的值,其他的保持不变。


    Args:
        boxitem (dict): 包含 box 的字典,格式为 {"box": [x1, y1, x2, y2], ...}
        key (str): box 的键名,默认为 "box"
        replace (str): polygon 的键名,默认为 "points"

    Returns:
         dict: 转换后的 polygon 表示,格式为 {"points": [[x1, y1], ..., [x1, y2]], ...}
    """
    box_item = deepcopy(boxitem)
    # 深拷贝,避免修改原始数据
    assert isinstance(box_item, dict), "box_item must be a dictionary"
    assert key in box_item, f"'{key}' not found in input"
    assert len(box_item[key]) == 4, f"'{key}' must contain four coordinates"

    x1, y1, x2, y2 = box_item[key]
    assert x1 < x2 and y1 < y2, f"Invalid box coordinates: {x1}, {y1}, {x2}, {y2}"

    # 这里主要是保持原来的数据结构, 只是替换key以及对应的值,其他的保持不变
    new_item = {}
    for k, v in box_item.items():
        if k == key:
            new_item[replace] = [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ]
        else:
            new_item[k] = v
    return new_item


def _polygon_to_box(
    polygonitem: dict, key: str = "points", replace: str = "box"
) -> dict:
    """将 polygon 表示（四个点）转换为 box 表示 [x1, y1, x2, y2]。


    polygon对应四个点来表示一个矩形框（左上,右上,右下,左下）. box利用左上角和右下角的坐标来表示一个矩形框。 二者进行互换, 只是替换key以及对应的值,其他的保持不变。

    Args:
        polygon_item (dict): 包含 polygon 的字典,例如 {"points": [[x1, y1], ..., [x1, y2]], ...}
        key (str): polygon 的键名,默认为 "points"
        replace (str): box 的键名,默认为 "box"

    Returns:
        dict: 转换后的 box 表示,例如 {"box": [x1, y1, x2, y2], ...}
    """
    polygon_item = deepcopy(polygonitem)
    # 深拷贝,避免修改原始数据

    assert isinstance(polygon_item, dict), "polygon_item must be a dictionary"
    assert key in polygon_item, f"'{key}' not found in input"

    points = polygon_item[key]
    assert isinstance(points, (list, tuple)) and len(points) == 4, (
        f"'{key}' must contain four points"
    )

    xs, ys = zip(*points, strict=False)
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    new_item = {}
    for k, v in polygon_item.items():
        if k == key:
            new_item[replace] = [int(x1), int(y1), int(x2), int(y2)]
        else:
            new_item[k] = v
    return new_item


def box_to_polygon(
    box: Union[dict, list], key: str = "box", replace: str = "points"
) -> Union[dict, list]:
    """将 box 格式 转换为 polygon 格式。

    !!! note
        - box 表示为 [x1, y1, x2, y2], 其中 x1, y1 为左上角坐标, x2, y2 为右下角坐标。

        - polygon 表示为 [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], 其中四个点分别表示左上、右上、右下、左下。

        polygon对应四个点来表示一个矩形框（左上,右上,右下,左下）,  box对应用两个点来表示一个矩形框（左上和右下）。 二者进行互换, 只是替换key以及对应的值,其他的保持不变。

    Args:
        box (Union[dict, list]): box 表示,
        key (str): box 的键名,默认为 "box"
        replace (str): polygon 的键名,默认为 "points", 可以和key相同,

    Returns:
        Union[dict, list]: 转换后的 polygon 格式

    Example:
        ```python
        from cfun.yolo.convert import box_to_polygon
        box = {"box": [10, 20, 30, 40], "cls": "cat", "conf": 0.9}
        polygon = box_to_polygon(box)
        print(polygon)
        # 输出: {"points": [[10, 20], [30, 20], [30, 40], [10, 40]], "cls": "cat", "conf": 0.9}

        box_list = [
            {"box": [10, 20, 30, 40], "cls": "cat", "conf": 0.9},
            {"box": [50, 60, 70, 80], "cls": "dog", "conf": 0.8}
        ]
        polygons = box_to_polygon(box_list)
        print(polygons)
        # 输出: [
        #     {"points": [[10, 20], [30, 20], [30, 40], [10, 40]], "cls": "cat", "conf": 0.9},
        #     {"points": [[50, 60], [70, 60], [70, 80], [50, 80]], "cls": "dog", "conf": 0.8}
        # ]
        # 注意: 这里的 box 表示是一个字典,包含了 box 的坐标、类别和置信度
        ```
    """
    if isinstance(box, dict):
        return _box_to_polygon(box, key, replace)
    elif isinstance(box, list):
        return [_box_to_polygon(b, key, replace) for b in box]


def polygon_to_box(
    polygon: Union[dict, list], key: str = "points", replace: str = "box"
) -> Union[dict, list]:
    """将 polygon 格式 转换为 box 格式。

    !!! note
        - box 表示为 [x1, y1, x2, y2], 其中 x1, y1 为左上角坐标, x2, y2 为右下角坐标。

        - polygon 表示为 [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], 其中四个点分别表示左上、右上、右下、左下。

        polygon对应四个点来表示一个矩形框（左上,右上,右下,左下）,  box对应用两个点来表示一个矩形框（左上和右下）。 二者进行互换, 只是替换key以及对应的值,其他的保持不变。

    Args:
        polygon (Union[dict, list]): polygon 表示
        key (str): polygon 的键名,默认为 "points"
        replace (str): box 的键名,默认为 "box", 可以和key相同.

    Returns:
        Union[dict, list]: 转换后的 box 格式

    Example:
        ```python
        from cfun.yolo.convert import polygon_to_box
        polygon = {"points": [[10, 20], [30, 20], [30, 40], [10, 40]], "name": "cat", "confidence": 0.9}
        box = polygon_to_box(polygon)
        print(box)
        # 输出: {"box": [10, 20, 30, 40], "name": "cat", "confidence": 0.9}

        polygon_list = [
            {"points": [[10, 20], [30, 20], [30, 40], [10, 40]], "name": "cat", "confidence": 0.9},
            {"points": [[50, 60], [70, 60], [70, 80], [50, 80]], "name": "dog", "confidence": 0.8}
        ]
        boxes = polygon_to_box(polygon_list)
        print(boxes)
        # 输出: [
        #     {"box": [10, 20, 30, 40], "name": "cat", "confidence": 0.9},
        #     {"box": [50, 60, 70, 80], "name": "dog", "confidence": 0.8}
        # ]
        ```
    """
    if isinstance(polygon, dict):
        return _polygon_to_box(polygon, key, replace)
    elif isinstance(polygon, list):
        return [_polygon_to_box(p, key, replace) for p in polygon]


if __name__ == "__main__":
    pass
    # json_to_yolo_txt(
    #     json_dir="imgsdata/xlabeljson",
    #     label_key="label",  # 指定分类的key
    #     class_mapping={0: "1"},
    #     output_dir="Label",  # 输出的txt文件的路径
    #     image_dir="imgsdata",
    #     image_suffix=".png",
    #     force_overwrite=True,
    #     shape_type="rectangle",
    # )
    # crop_images(
    #     json_dir="imgsdata/xlabeljson",
    #     image_dir="imgsdata",
    #     category_key="description",
    #     output_dir="cropped",  # 输出的裁剪图片的路径
    #     force_overwrite=True,  # 是否强制覆盖输出目录
    # )

    # 旋转矩阵
    # json_to_yolo_txt(
    #     json_dir="weilai1_rotation/weilai1_json_rotated",
    #     label_key="label",  # 指定分类的key
    #     class_mapping={0: "fan", 1: "zheng", 2: "touxiang", 3: "guohui"},
    #     output_dir="Label_rotated",  # 输出的txt文件的路径
    #     image_dir="weilai1_rotation",
    #     image_suffix=".jpg",
    #     force_overwrite=True,  # 是否强制覆盖输出目录
    #     shape_type="rotation",
    # )

    # crop_images(
    #     json_dir="weilai1_rotation/weilai1_json_rotated",
    #     image_dir="weilai1_rotation",
    #     image_suffix=".jpg",
    #     category_key="label",
    #     isremove_chinese=True,
    #     output_dir="cropped_rotated",  # 输出的裁剪图片的路径
    #     force_overwrite=True,  # 是否强制覆盖输出目录
    #     shape_type="rotation",
    # )
