import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from PIL import Image


class XLabel:
    def __init__(
        self,
        image_path: str | Path,
        data: list[dict],
        datakey: str = "points",
        platform: str = "",
        fixedtimestamp: bool = False,
        filemd5: Optional[str] = None,
        namereplace: Optional[dict] = None,
        shape_type: str = "rectangle",
    ) -> None:
        """初始化XLabel类

        XLabel类用于生成x-anylabeling格式的json文件

        Args:
            image_path (str): 图片路径, 必须真是存在的文件路径
            data (list[dict]): 标记的坐标,且每个dict中必须包含points这个key, 这个ponits是一个列表,包含四个点的坐标,eg：[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            datakey (str): 标记的坐标点对应的key, 默认是 points, 这个key必须在data中存在, 否则会报错.
            platform (str): 平台名称, 自定义, 默认空字符串
            fixedtimestamp (bool): 是否固定时间戳,默认False, 如果为True,则时间戳为0,否则为当前时间戳
            filemd5 (Optional[str]): 文件的md5值,默认None, 会根据文件自动计算md5值
            namereplace (dict): 替换名称的方式,是一个字典, 其中的key是data中的key,value是模板中shapes字段下的key, 用data中的key来替代模板中的key, 如果没有则不传递
            shape_type (str): 形状类型,默认rectangle, 暂时可选 rectangle, rotation.

        !!! note
            shape_type 目前只支持rectangle, rotation.

            - `rectangle`: 矩形
            - `rotation`: 旋转矩形, 需要格外的信息,即旋转角度, 需要在data中添加一个key, 然后再namereplace中添加一个映射值为 direction即可,


        Example:
            ```python
            from cfun.yolo.xlabel import XLabel
            image_path = "tests/images/image_detect_01.png"

            ### 矩形的例子
            data = [
                {
                    "name": "char1",
                    "points": [[100, 200], [200, 200], [200, 300], [100, 300]],
                    "confidence": 0.9,
                },
                {
                    "name": "char2",
                    "points": [[300, 400], [400, 400], [400, 500], [300, 500]],
                    "confidence": 0.8,
                },
            ]

            xl = XLabel(
                image_path,
                data,
                datakey="points",  # 这个key必须在data中存在
                platform = "yolo",
                fixedtimestamp=True,
                namereplace={"name": "description"},
            )
            # 建议json的名字和图片的名字一致(这里是为了测试)
            xl.save_template("output/template1.json")  # 输出json文件

            #### 旋转矩形的例子
            image_path = "tests/images/image_detect_01.png"
            data = [
                {
                    "name": "char1",
                    "points": [[100, 200], [200, 200], [200, 300], [100, 300]],
                    "confidence": 0.9,
                    "angle": 45,  # 旋转角度
                },
                {
                    "name": "char2",
                    "points": [[300, 400], [400, 400], [400, 500], [300, 500]],
                    "confidence": 0.8,
                    "angle": 25,
                },
            ]
            namereplace = {
                "name": "description",
                "angle": "direction",  # 这里需要添加一个映射值为 direction的映射值
            }
            xl = XLabel(
                image_path,
                data,
                datakey="points",  # 这个key必须在data中存在
                platform = "yolo",
                fixedtimestamp=True,
                shape_type="rotation",
                namereplace=namereplace,
            )
            xl.save_template("output/template2.json")
            ```

        """
        if isinstance(image_path, (str, Path)):
            image_path = Path(image_path)

        if isinstance(data, list):
            data = deepcopy(data)
        self.datakey = datakey
        assert image_path.exists(), f"image_path: {image_path} 不存在"
        assert image_path.is_file(), f"image_path: {image_path} 不是文件"
        assert image_path.suffix.lower() in [".png", ".jpg", ".jpeg"], (
            f"image_path: {image_path} 不是图片文件"
        )
        assert self._check_data(data, namereplace if namereplace is not None else {})
        assert isinstance(fixedtimestamp, bool), (
            f"fixedtimestamp: {fixedtimestamp} 不是布尔值"
        )
        assert filemd5 is None or isinstance(filemd5, str), (
            f"filemd5: {filemd5} 不是字符串"
        )
        assert isinstance(platform, str), f"platform: {platform} 不是字符串"

        self.image_path = image_path
        self.data = data

        self.shape_type = shape_type
        if shape_type not in ["rectangle", "rotation"]:
            raise ValueError(
                f"shape_type: {shape_type} 不是 rectangle 或 rotation, 目前只支持这两种类型"
            )
        if shape_type == "rotation":
            assert namereplace is not None and "direction" in namereplace.values(), (
                f"shape_type: {shape_type} 时需要在namereplace中添加一个value为 direction的映射值"
            )

        self.platform = platform
        self.fixedtimestamp = fixedtimestamp
        self.filemd5 = filemd5 if filemd5 else self._calculate_md5()
        self.namereplace = namereplace
        self.image_width, self.image_height, _ = self._get_image_size()
        self.imagename = self.image_path.name
        self.imagesuffix = self.image_path.suffix

    def _check_data(self, data: list[dict], namereplace: dict) -> bool:
        """
        检查数据是否符合要求
        :param data: 数据
        :param namereplace: 替换名称的方式,是一个字典, 其中的key是data中的key,value是模板中的key, 用data中的key来替代模板中的key, 如果没有则不传递
        :return: 是否符合要求
        """
        if namereplace:
            assert isinstance(namereplace, dict), f"namereplace: {namereplace} 不是字典"
            # key不应该有重复
            assert len(namereplace) == len(set(namereplace.keys())), (
                f"namereplace: {namereplace} 有重复的key"
            )
        assert isinstance(data, list), "参数 data 不是列表"
        assert len(data) > 0, "参数 data 不能为空列表"
        assert all(isinstance(item, dict) for item in data), "参数 data 不是字典列表"

        assert all(
            isinstance(item[self.datakey], list)
            and len(item[self.datakey]) == 4
            and all(
                isinstance(point, list) and len(point) == 2
                for point in item[self.datakey]
            )
            for item in data
        ), (
            f"参数 data 中的 {self.datakey} 不是列表,或者长度不为4,或者每个点不是列表,或者长度不为2"
        )

        # 如果传递了namereplace, 则检查namereplace中的key是否在data中存在
        if not namereplace:
            return True

        # 所有的key 都应该在data中存在
        for key in namereplace.keys():
            for item in data:
                if key not in item:
                    raise ValueError(f"参数data中不包含key: {key}")

        return True

    def _get_image_size(self) -> tuple[Any, Any, Any]:
        """获取图片的宽高和通道数"""
        image_path = str(self.image_path)
        with Image.open(image_path) as img:
            width, height = img.size
        return width, height, None

    def _calculate_md5(self) -> str:
        """计算文件的 MD5 值"""
        file_path = self.image_path
        hash_md5 = hashlib.md5()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _obtain_attributes(self) -> dict:
        """
        获取模板的属性
        :return: 属性字典
        """
        return {
            "pingtai": self.platform,  # 平台名称
            "timestamp": 0 if self.fixedtimestamp else int(time.time()),  # 时间戳
            "rawimgmd5": self.filemd5,  # 文件的 md5 值
        }

    def _obtain_shapes(self) -> list[dict]:
        oneshape = {
            "kie_linking": [],
            # 标签的类别, 也可以用data中的name来替代
            "label": None,  # 必须是字符串
            # 置信度, 可选, 可以用data中的confidence来替代
            "score": None,  # 置信度, 可选（可以是数字）
            # 坐标点, 需要是一个列表, 包含四个点的坐标,
            "points": [],  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            "group_id": None,  # 组ID, 可选
            # 描述信息,对应data中的name
            "description": "",
            "difficult": False,
            "shape_type": self.shape_type,  # 形状类型, 默认rectangle, 可选 rectangle, rotation
            "flags": {},
            "attributes": {},
        }
        shapes = []
        for item in self.data:
            # 复制模板
            shape = deepcopy(oneshape)
            # 获取描述信息
            if self.namereplace:
                for key, value in self.namereplace.items():
                    if key in item and value in shape:
                        shape[value] = item[key]
            # 填充数据
            shape["points"] = item[self.datakey]
            shape["attributes"] = self._obtain_attributes()
            # 保持字符串格式
            if not isinstance(shape["description"], str):
                shape["description"] = str(shape["description"])

            if not isinstance(shape["label"], str):
                shape["label"] = str(shape["label"])

            # 保留两位小数
            if isinstance(shape["score"], (int, float)):
                shape["score"] = round(shape["score"], 2)

            # 置信度
            shapes.append(shape)
        return deepcopy(shapes)

    def _generate_template(self, version="2.5.4") -> dict:
        # 填充模板数据
        template_data = {
            "version": version,  # 固定版本
            "flags": {},  # 默认无标记
            "shapes": self._obtain_shapes(),  # 目标框数据
            "imagePath": self.imagename,  # 图片名称
            "imageData": None,  # 默认无图像数据
            "imageHeight": self.image_height,  # 图片高度
            "imageWidth": self.image_width,  # 图片宽度
        }
        return deepcopy(template_data)

    def save_template(self, save_path: str | Path, version="2.5.4") -> None:
        """保存模板数据到文件

        Args:
            save_path (str | Path): 保存的文件路径
            version (str): 模板版本, 默认是2.5.4
        """
        template_data = self._generate_template(version=version)
        with open(str(save_path), "w", encoding="utf-8") as f:
            json.dump(template_data, f, indent=2, ensure_ascii=False)

    def to_json(self, save_path: str | Path, version="2.5.4") -> None:
        """保存模板数据到文件(同 save_template 方法)

        Args:
            save_path (str | Path): 保存的文件路径
            version (str): 模板版本, 默认是2.5.4
        """
        self.save_template(save_path, version=version)
