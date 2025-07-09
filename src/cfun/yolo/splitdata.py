import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Any, Optional, Sequence, Tuple

from jsonpath import jsonpath

# from sklearn.model_selection import train_test_split
from .convert import json_to_yolo_txt


def train_test_split(
    *arrays: Sequence[Any],
    test_size: Optional[float] = None,
    train_size: Optional[float] = None,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: Optional[Sequence] = None,
) -> Tuple:
    if not arrays:
        raise ValueError("At least one array must be provided")

    n_samples = len(arrays[0])
    for arr in arrays:
        if len(arr) != n_samples:
            raise ValueError("All input arrays must have the same length")

    if test_size is None and train_size is None:
        test_size = 0.25
    if test_size is not None:
        if isinstance(test_size, float):
            n_test = int(n_samples * test_size)
        else:
            n_test = test_size
    else:
        if isinstance(train_size, float):
            n_test = n_samples - int(n_samples * train_size)
        elif train_size is not None:
            n_test = n_samples - train_size
        else:
            raise ValueError("train_size must not be None when test_size is None")

    n_train = n_samples - n_test

    rng = random.Random(random_state)

    if stratify is not None:
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(stratify):
            label_to_indices[label].append(idx)

        train_idx, test_idx = [], []
        for _, indices in label_to_indices.items():
            rng.shuffle(indices)
            split_point = int(len(indices) * n_train / n_samples)
            train_idx.extend(indices[:split_point])
            test_idx.extend(indices[split_point:])
    else:
        indices = list(range(n_samples))
        if shuffle:
            rng.shuffle(indices)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

    def split(arr):
        return [arr[i] for i in train_idx], [arr[i] for i in test_idx]

    result = []
    for arr in arrays:
        result.extend(split(arr))
    return tuple(result)


def _create_directories(base_imgpath: str | Path, base_txtpath: str | Path):
    """Create the necessary directories for train, val, and test sets."""
    Path(base_imgpath).mkdir(parents=True, exist_ok=True)
    Path(base_txtpath).mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        os.makedirs(Path(base_imgpath) / split, exist_ok=True)
        os.makedirs(Path(base_txtpath) / split, exist_ok=True)


def _copy_files(
    file_list: list[str],
    imgpath: str | Path,
    txtpath: str | Path,
    new_imgpath: str | Path,
    new_txtpath: str | Path,
    postfix: str,
    subdir: str,
):
    """Copy image and label files for each split (train, val, test)."""
    for file in file_list:
        try:
            img_source = Path(imgpath) / f"{file}{postfix}"
            txt_source = Path(txtpath) / f"{file}.txt"
            assert img_source.exists(), f"Image file {img_source} does not exist."
            assert txt_source.exists(), f"Label file {txt_source} does not exist."

            img_dest = Path(new_imgpath) / subdir / f"{file}{postfix}"
            txt_dest = Path(new_txtpath) / subdir / f"{file}.txt"

            shutil.copy2(img_source, img_dest)
            shutil.copy2(txt_source, txt_dest)
        except Exception as e:
            print(f"Error copying {file}: {e}")


def splitdata(
    imgpath: str | Path,
    txtpath: str | Path,
    new_imgpath: str | Path,
    new_txtpath: str | Path,
    val_size: float = 0.1,
    test_size: float = 0.1,
    postfix: str = ".png",
) -> None:
    """将数据集拆分为训练集、val集和测试集,并相应地复制文件。(主要用于yolo目标检测数据的划分)

    主要用于yolo目标检测数据的划分,分成训练集、验证集和测试集。

    !!! note
        对于分类数据, 官方提供了另外的函数可以直接划分
        ```python
        from ultralytics.data.split import split_classify_dataset
        ```

    Args:
        imgpath (str | Path): 原始图片路径的根目录 (这个目录下包含了要处理的图片)
        txtpath (str | Path): 原始标签路径的根目录 (txt文件, 图片文件和txt文件的stem要一样,没有重复样本且数量也要一致,否则可能报错)
        new_imgpath (str | Path): 新的图片路径
        new_txtpath (str | Path): 新的标签路径
        val_size (float, optional): 验证集所占比例. Defaults to 0.1.
        test_size (float, optional): 测试集所占比例. Defaults to 0.1.
        postfix (str, optional): 图片后缀名. Defaults to ".png".


    Raises:
        AssertionError: 如果原始图片路径或标签路径不存在,抛出异常。
        AssertionError: 如果txt文件和图片文件的stem不一致 或者数量不一致,抛出异常。
        ValueError: 如果val_size和test_size不在0到1之间,抛出异常。
        ValueError: 如果val_size + test_size >= 1,抛出异常。

    Example:
        ```python
        from cfun.yolo.splitdata import splitdata
        imgpath = "imgsdata"  #图片的路径
        txtpath = "detect"  #标签的路径
        new_imgpath = "./imgs_split/train/images"  #新的图片路径
        new_txtpath = "./imgs_split/train/labels" #新的标签路径
        splitdata(imgpath, txtpath, new_imgpath, new_txtpath)
        ```
    """
    if not (0 <= val_size <= 1):
        raise ValueError("val_size must be between 0 and 1")
    if not (0 <= test_size <= 1):
        raise ValueError("test_size must be between 0 and 1")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be less than 1")
    if isinstance(imgpath, str):
        imgpath = Path(imgpath)
    if isinstance(txtpath, str):
        txtpath = Path(txtpath)
    if isinstance(new_imgpath, str):
        new_imgpath = Path(new_imgpath)
    if isinstance(new_txtpath, str):
        new_txtpath = Path(new_txtpath)

    assert imgpath.is_dir() and txtpath.exists(), (
        f"Image path {imgpath} is not a directory or label path {txtpath} does not exist."
    )
    assert txtpath.is_dir() and imgpath.exists(), (
        f"Label path {txtpath} is not a directory or image path {imgpath} does not exist."
    )

    #  计算验证集所占的比例
    _val_size = val_size / (1 - test_size)

    # 创建必要的目录
    _create_directories(new_imgpath, new_txtpath)

    # 遍历txtpath 下的所有txt
    names = [i.stem for i in txtpath.glob("*.txt") if i.is_file()]

    # names 未重复
    assert len(names) == len(set(names)), (
        f"Label files in {txtpath} are not unique. Please check the files."
    )

    train, test = train_test_split(
        names, test_size=test_size, shuffle=True, random_state=0
    )
    train, val = train_test_split(
        train, test_size=_val_size, shuffle=True, random_state=0
    )

    s0 = f"train set size: {len(train)} val set size: {len(val)} test set size: {len(test)}"
    print(s0)

    # Copy the files to the appropriate directories
    _copy_files(
        train, imgpath, txtpath, new_imgpath, new_txtpath, postfix, subdir="train"
    )
    _copy_files(val, imgpath, txtpath, new_imgpath, new_txtpath, postfix, subdir="val")
    _copy_files(
        test, imgpath, txtpath, new_imgpath, new_txtpath, postfix, subdir="test"
    )

    return None


def check_image_and_json(
    json_dir: Path | str, label_key: str, image_dir: Path | str, image_suffix: str
) -> dict:
    """检查json_dir下的json文件 和 image_dir下的图片文件是否一致,并返回类别映射文件

    !!! note
        检查三个点:

        - json_dir下的json文件, image_dir下的图片文件, 以及json文件中的imagePath字段,这三者的 stem 数量和名字要相同,不能重复.

    Args:
        json_dir (Path | str): JSON 标注文件所在目录
        label_key (str): 用于分类的键, 如 "label", 这个key在json文件中的shapes字段下才行
        image_dir (Path | str): 图像文件所在目录,
        image_suffix (str): 图像文件后缀名,

    Raises:
        AssertionError: 如果json_dir或image_dir不是有效的目录,抛出异常。
        AssertionError: 如果json_dir下的json文件数量和image_dir下的图片文件数量不一致,抛出异常。
        AssertionError: 如果json文件的stem和图片文件的stem不一致,抛出异常。
        AssertionError: 如果json文件中的imagePath字段和图片文件名不一致,抛出异常。

    Returns:
        dict: 类别映射文件,格式为 `{index: class_name}`, 这里的index是从0开始的, class_name是类别名称,对应label_key

    Example:
        ```python
        from cfun.yolo.splitdata import check_image_and_json
        json_dir = "imgs/xlabel"
        label_key = "label"
        image_dir = "imgs"
        image_suffix = ".jpg"
        class_mapping = check_image_and_json(json_dir, label_key, image_dir, image_suffix)
        print(class_mapping)
        ```
    """
    json_dir = Path(json_dir)
    image_dir = Path(image_dir)
    assert json_dir.is_dir(), f"json_dir {json_dir} is not a valid directory"
    assert image_dir.is_dir(), f"image_dir {image_dir} is not a valid directory"
    assert image_suffix.startswith("."), f"Invalid image suffix: {image_suffix}"

    ##### 需要对 json_dir下的json文件 和 image_dir下的图片文件 进行三次检查 #####
    ##3 1. 检查json_dir下的json文件 和 image_dir下的图片 数量是否一致
    json_files = sorted([f for f in json_dir.glob("*.json") if f.is_file()])
    image_files = sorted([f for f in image_dir.glob(f"*{image_suffix}") if f.is_file()])
    # 由于后缀名大写和小写的问题,有些时候要小心

    assert len(json_files) == len(image_files), (
        f"检查不通过, json_dir下的json文件数量和image_dir下的图片文件数量不一致,json_files数量: {len(json_files)}, image_files数量: {len(image_files)}"
    )

    ### 2. 提取json文件的stem和图片文件的stem,检查是否一致, 且不能重复
    json_stems = {f.stem for f in json_files}
    image_stems = {f.stem for f in image_files}
    assert json_stems == image_stems, (
        f"检查不通过, json文件的stem和图片文件的stem不一致, json_stems数量: {len(json_stems)}, image_stems数量: {len(image_stems)}"
    )

    ### 3. 读取json文件并提取其中的imagePath字段,然后提取name, 应该和图片文件一致
    image_names = {f.name for f in image_files}
    labels = set()
    image_names_in_json = set()
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_path = jsonpath(data, "$..imagePath")
        assert image_path, f"Missing imagePath in {jf}"
        image_name = Path(image_path[0]).name
        image_names_in_json.add(image_name)
        shapes = data.get("shapes", [])
        label_values = jsonpath(shapes, f"$..{label_key}")
        if label_values:
            labels.update(label_values)
        else:
            raise ValueError(f"Missing label key '{label_key}' in {jf}")

    diffset1 = image_names_in_json - image_names
    diffset2 = image_names - image_names_in_json
    if len(diffset1) > 0:
        raise AssertionError(
            f"检查不通过, json文件中的imagePath字段包含了不存在的图片文件名(前 5 个, 总数: {len(diffset1)}):\n{list(diffset1)[:5]}"
        )
    if len(diffset2) > 0:
        raise AssertionError(
            f"检查不通过, 图片文件名包含了json文件中的imagePath字段不存在的图片文件名(前 5 个, 总数: {len(diffset2)}):\n{list(diffset2)[:5]}"
        )
    assert image_names_in_json == image_names, (
        f"检查不通过, json文件中的imagePath字段和图片文件名不一致, json内部的imagePath数量: {len(image_names_in_json)}, image_names数量: {len(image_names)}"
    )

    # 生成有序类别映射
    return {i: label for i, label in enumerate(sorted(labels))}


def _mkcurrent_temp_dir(base_dir: str) -> Path:
    """创建一个唯一的临时目录,避免与已存在的目录冲突"""
    base = Path(base_dir)
    for i in range(10):  # 避免死循环
        candidate = base if i == 0 else Path(f"{base_dir}_{i}")
        # 如果这个目录为空目录,则直接返回这个目录
        if candidate.exists() and candidate.is_dir() and not any(candidate.iterdir()):
            return candidate

        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise RuntimeError("无法创建唯一的临时目录")


def json_to_yolo_detect_data(
    json_dir: Path | str, label_key: str, image_dir: Path | str, image_suffix: str
) -> None:
    """将json数据直接转为yolo训练数据集,且已划分为训练集、验证集和测试集。主要用于yolo目标检测数据的划分(一步到位版本)

    会在当且文件夹下创建一个imgs_split的文件夹,里面包含了train、val和test三个文件夹,分别对应训练集、验证集和测试集。


    Args:
        json_dir (Path | str): JSON 标注文件所在目录
        label_key (str): 用于分类的键,如 "label",这个key在json文件中的shapes字段下才行
        image_dir (Path | str): 图像文件所在目录,
        image_suffix (str): 图像文件后缀名,

    Example:
        ```python
        from cfun.yolo.splitdata import json_to_yolo_detect_data
        json_to_yolo_detect_data(
            json_dir="imgs/xlabel",
            label_key="label",
            image_dir="imgs",
            image_suffix=".jpg",
        )
        ```

    """
    assert Path(json_dir).is_dir(), f"json_dir {json_dir} is not a directory"
    assert Path(image_dir).is_dir(), f"image_dir {image_dir} is not a directory"
    temp_dir = _mkcurrent_temp_dir("temp")

    class_mapping = check_image_and_json(
        json_dir=json_dir,
        label_key=label_key,
        image_dir=image_dir,
        image_suffix=image_suffix,
    )
    print("通过检查,类别映射文件: ")
    pprint(class_mapping)
    json_to_yolo_txt(
        json_dir=json_dir,
        label_key=label_key,
        class_mapping=class_mapping,
        output_dir=temp_dir,
        image_dir=image_dir,
        image_suffix=image_suffix,
        force_overwrite=True,
        ischeck=True,
    )
    # 创建新的目录---作为最后的输出目录
    outputdir = _mkcurrent_temp_dir("imgs_split")

    splitdata(
        imgpath=image_dir,
        txtpath=temp_dir,
        new_imgpath=str(outputdir / "train" / "images"),
        new_txtpath=str(outputdir / "train" / "labels"),
        val_size=0.1,
        test_size=0.1,
        postfix=image_suffix,
    )
    # 把 class_mapping 保存到新的目录下
    with open(outputdir / "class_mapping.txt", "w", encoding="utf-8") as f:
        for index, class_name in class_mapping.items():
            f.write(f"{index}: {class_name}\n")
    # 删除临时目录
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # imgpath = "imgsdata"  # 图片的路径
    # txtpath = "detect"  # 标签的路径
    # new_imgpath = "./imgs_split/train/images"  # 新的图片路径
    # new_txtpath = "./imgs_split/train/labels"  # 新的标签路径
    # splitdata(imgpath, txtpath, new_imgpath, new_txtpath)

    json_to_yolo_detect_data(
        json_dir="imgs/xlabel",
        label_key="label",
        image_dir="imgs",
        image_suffix=".jpg",
    )
