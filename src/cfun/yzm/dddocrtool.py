"""
ddddocr 目标检测工具类

这个类主要是对 ddddocr 进行封装, 方便使用

主要的功能：

1. 目标检测,获取目标检测框

2. 获取图片的宽高

3. 将 ddddocr 返回的目标检测框转换为 xlabing 的格式 (xyxy格式)

4. 删除太近边缘的框

5. 检查坐标是否在给定的矩形范围内

6. 扩展超级鹰识别的结果,增加目标检测框的坐标 （主要的功能）

7. 检查框的格式是否正确



"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Optional

import ddddocr
from PIL import Image


class ImageDet:
    def __init__(self):
        self.det = ddddocr.DdddOcr(det=True, show_ad=False)

    def detection(self, image_path: str):
        with open(image_path, "rb") as f:
            image = f.read()
        bboxes = self.det.detection(image)
        # 返回的是一个list, 每个元素是一个框的坐标
        # eg: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...] ,xyxy格式
        return bboxes

    def get_image_size(self, image_path: str | Path) -> tuple[int, int, None]:
        """通过 PIL 库获取图片的宽高

        Args:
            image_path (str | Path): 图片路径

        Returns:
            tuple(width, height, channels):  图片的宽高和通道数
        """
        if isinstance(image_path, Path):
            image_path = str(image_path)

        with Image.open(image_path) as img:
            width, height = img.size
        return width, height, None

    def toxlabeling(
        self, image_path: Optional[str] = None, bboxes: Optional[list] = None
    ) -> list:
        """把 ddddocr 返回的目标检测框转换为 xlabing 的格式 (xyxy格式)

        Args:
            image_path (str): 图片路径 (与 bboxes 二选一)
            bboxes (list): [[x1, y1, x2, y2], ...] （与 image_path 二选一）, 这个坐标 左上,右下, 也就是 xyxy 的格式

        Returns:
            list: [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]], ...]

        PS:
            1. 这个函数主要是把 ddddocr 返回的目标检测框转换为 xlabing 的格式, 也就是四个点的格式, 具体的格式是：

            ```
            [
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                ...
            ]
            ```
        """
        if image_path is None and bboxes is None:
            raise ValueError("image_path 和 bboxes 必须二选一,不能都为 None")

        if image_path is not None and bboxes is not None:
            raise ValueError("image_path 和 bboxes 不能同时使用")

        if bboxes is not None:
            bboxes = self.detection(image_path)

        assert isinstance(bboxes, list)
        assert all(isinstance(bbox, list) and len(bbox) == 4 for bbox in bboxes)

        result = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            result.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        return result

    @staticmethod
    def _delete_bianyuan(bboxes, image_width, image_height, b=5):
        """删除太近边缘的框 (xyxy)

        Args:
            bboxes (list): [[x1, y1, x2, y2], ...], （左上角, 右下角）,ddddocr 返回的格式
            image_width (int): 图片宽度
            image_height (int): 图片高度
            b (int): 边缘距离, 默认是 5

        Returns:
            list: 新的框列表,还是 [[x1, y1, x2, y2], ...] 的格式
        """
        assert all(isinstance(bbox, list) and len(bbox) == 4 for bbox in bboxes), (
            "bboxes 中的每个元素必须是一个列表且长度为4"
        )
        newbboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # 检查框是否在边缘
            if x1 < b or y1 < b or x2 > image_width - b or y2 > image_height - b:
                continue
            newbboxes.append(bbox)
        bboxes = newbboxes
        return bboxes

    @staticmethod
    def _check_range(centerxy, box) -> bool:
        """检查坐标是否在给定的矩形范围内

        Args:
            centerxy (list): 坐标 [x, y]
            box (list): 矩形框的坐标 [x1, y1, x2, y2] （左上角, 右下角）,ddddocr 返回的格式

        Returns:
            bool: True or False
        """
        assert len(box) == 4, "box must be a list of 4 points"
        assert len(centerxy) == 2, "corxy must be a list of 2 points"
        assert isinstance(box, list), "box must be a list"
        assert isinstance(centerxy, list), "corxy must be a list"
        minx = min(box[0], box[2])
        maxx = max(box[0], box[2])
        miny = min(box[1], box[3])
        maxy = max(box[1], box[3])
        x, y = centerxy

        if minx <= x <= maxx and miny <= y <= maxy:
            # 判断坐标是否在矩形范围内
            return True
        else:
            return False

    def extendCJYRecognition(self, ocrdata: list, image_path: str) -> list:
        """扩展超级鹰识别的结果. (也可以是其他平台的结果, 因为很多平台返回的结果都只含中心点坐标)

        Args:
            ocrdata (list of dict): 超级鹰识别的结果,每个元素是一个字典,格式如下：

                - name (str): 字符内容,例如 `"之"`。
                - coordinates (list of int): 坐标列表,格式如 `[x, y]`。

            image_path (str): 图片路径。

        Returns:
            list of dict: 扩展后的结果,每个字典包含以下字段：

                - name (str): 字符内容。
                - coordinates (list of int): 原始坐标。
                - points (list of list of float): 四个顶点坐标,格式如 `[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]`。
                - image_width (int): 图片宽度。
                - image_height (int): 图片高度。

        Example:
            ```python

            '''
            超级鹰识别的结果是一个： '之,207,115|成,158,86|人,126,44'
                我们可以很轻松的转为：
                [
                    {
                        "name": "之",
                        "coordinates": [207, 115]
                    },
                    {
                        "name": "成",
                        "coordinates": [158, 86]
                    },
                    {
                        "name": "人",
                        "coordinates": [126, 44]
                    }
                ]
            以上只是返回中心点的坐标, 但是我希望得到目标检测框的坐标,即 四个定点的坐标,
            因此在这个结果的基础上,增加一个key, 叫做 points, 其值为 [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            采用 xyxy 的格式, 这样就可以直接使用了. 这个目标检测的框如何产生呢？ 这里采用ddddocr的目标检测框来进行扩展。
            '''
            from cfun.yzm.dddocrtool import ImageDet
            imgdet = ImageDet()
            ocrdata = [
                {"name": "之", "coordinates": [207, 115]},
                {"name": "成", "coordinates": [158, 86]},
                {"name": "人", "coordinates": [126, 44]}
            ]
            image_path = "restoredUnique2/0a19bbddea_乏宙泡瓜色.png"
            extended_data = imgdet.extendCJYRecognition(ocrdata, image_path)
            print(f"Extended Data: {extended_data}")
            '''
            [
                {
                    "name": "之",
                    "coordinates": [207, 115],
                    "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    "image_width": 100,
                    "image_height": 100
                },
                {
                    "name": "成",
                    "coordinates": [158, 86],
                    "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "image_width": 100,
                    "image_height": 100
                },
                {
                    "name": "人",
                    "coordinates": [126, 44],
                    "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "image_width": 100,
                    "image_height": 100
                }
            ]
            '''
            ```
        """
        bboxes = self.detection(
            image_path
        )  # 返回的坐标,[[x1, y1, x2, y2], [x1, y1, x2, y2], ...] ,xyxy格式
        # 删除一些检测有问题的框, 比如框距离边缘太近的框
        image_width, image_height, _ = self.get_image_size(image_path)
        bboxes = self._delete_bianyuan(bboxes, image_width, image_height)

        # 这里需要判断一下, 如果框的中心点在框内, 则返回这个框的坐标
        for box in bboxes:
            x1, y1, x2, y2 = box
            for idata in ocrdata:
                xy = idata["coordinates"]
                if self._check_range(xy, box):
                    # 如果在框内, 则返回这个框的坐标
                    idata["points"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    idata["image_width"] = image_width
                    idata["image_height"] = image_height
                    break
        # 移除没有 points 的框
        ocrdata = [idata for idata in ocrdata if "points" in idata]
        return deepcopy(ocrdata)

    @staticmethod
    def check_bboxes_four(bboxes: list) -> bool:
        """检查框的格式是否正确(四个点的格式)

        Args:
            bboxes (list): [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]], ....]

        Returns:
            bool: True or False
        """
        if not isinstance(bboxes, list):
            return False
        if all(
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(point, list) and len(point) == 2 for point in bbox)
            for bbox in bboxes
        ):
            return True
        else:
            return False

    @staticmethod
    def check_bboxes_two(bboxes: list) -> bool:
        """检查框的格式是否正确（两个点的格式）

        Args:
            bboxes (list): [[x1, y1, x2, y2], ....]

        Returns:
            bool: True or False
        """
        if all(isinstance(bbox, list) and len(bbox) == 4 for bbox in bboxes):
            return True
        else:
            return False


if __name__ == "__main__":
    imgdet = ImageDet()
    image_path = "assets/image_detect_01.png"
    ## 假设json中存储的就是超级鹰的结果（且已经转为 [{"name": "之", "coordinates": [207, 115]}, ....] 的json格式）
    ## 二者要对应的上
    resfile = "yolo_response/0a19bbddea_乏宙泡瓜色.json"
    with open(resfile, "r", encoding="utf-8") as f:
        resdata = json.load(f)

    extended_data = imgdet.extendCJYRecognition(resdata, image_path)
    print(f"Extended Data: {extended_data}")
