"""
主要是根据 yolo11的目标检测和分类模型,对图片进行检测和分类,返回一个list[dict],每个元素是一个字典,包含名称和坐标, 依赖 ultralytics

1. 首先通过检测模型,得到检测框的坐标
2. 然后根据检测框的坐标,裁剪出图片
3. 然后通过分类模型,对裁剪出来的图片进行分类
4. 最后将分类结果和坐标一起返回, 返回一个list[dict]


"""

from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from .base import load_image
from .classify import Classifier
from .detect import Detector


class DetClsOnnx:
    """YOLO模型类,用于目标检测和分类, 传入的是onnx模型.  只适用于一张图上同时有检测和分类的情况

    Attributes:
        det (YOLO): YOLO检测模型
        cls (YOLO): YOLO分类模型
    """

    def __init__(
        self,
        det_model: None | str | Path = None,
        cls_model: None | str | Path = None,
        det_imgsz: int | tuple[int, int] | list[int] | None = (640, 640),
        cls_imgsz: int | tuple[int, int] | list[int] | None = (64, 64),
    ):
        """初始化YOLO模型

        Args:
            det_model (Optional[str]): 检测模型路径,默认为None,使用默认模型
            cls_model (Optional[str]): 分类模型路径,默认为None,使用默认模型
            det_imgsz (int): 检测模型输入图像大小,默认为640
            cls_imgsz (int): 分类模型输入图像大小,默认为64
        """
        if det_model is None and cls_model is None:
            from cfundata import cdata

            det_model = cdata.DX_DET_ONNX
            cls_model = cdata.DX_CLS_ONNX

        # 加载模型
        self.det = Detector(
            model=self._resolve_path(det_model, "Detection"),
            imgsz=det_imgsz,
        )

        self.cls = Classifier(
            model=self._resolve_path(cls_model, "Classification"),
            imgsz=cls_imgsz,
        )

    @staticmethod
    def _resolve_path(path, name):
        assert path, f"{name} model path is None"
        path = str(path) if isinstance(path, Path) else path
        path_obj = Path(path)

        assert path_obj.exists(), f"{name} model not found: {path}"
        assert path_obj.suffix == ".onnx", f"{name} model must be a .onnx file"

        return path

    def predict(self, source: Union[str, Path, Image.Image, np.ndarray]) -> list[dict]:
        """根据输入的图片路径进行检测和分类,并返回结果 (单个图片)

        Args:
            source (Union[str, Path, Image.Image, np.ndarray]): 输入的图片路径或图片对象

        Returns:
            list[dict]: 检测和分类结果, 每个元素是一个字典,包含名称和坐标等详细信息

                - `box` (list[float]): 检测框的坐标,box 格式。
                - `conf` (float): 检测框的置信度。
                - `cls` (int): 检测框的类别索引。
                - `name` (str): 检测框的名称。
                - `points` (list[list[float]]): 检测框的多边形坐标。
                - `top1name` (str): 分类结果的名称（Top-1）。
                - `top1conf` (float): 分类结果的置信度（Top-1）。
                - `top1` (int): 分类结果的索引（Top-1）。
                - `top5name` (list[str]): 前5个分类结果的名称。
                - `top5conf` (list[float]): 前5个分类结果的置信度。
                - `top5` (list[int]): 前5个分类结果的索引。
                - `orig_shape` (tuple[int, int]): 原始图片的大小。 (宽, 高)

        Example:
            ```python
            from cfundata import cdata
            from PIL import ImageDraw, ImageFont
            from cfun.yolo.detclsonnx import DetClsOnnx

            yolo = DetClsOnnx()
            image_path = "assets/image_detect_01.png"
            results = yolo.predict(image_path)
            print(results)
            '''
            #输出类似: (每一个框都是一个字典,包含名称和坐标等相关信息
                [
                    {
                        ############### 下面是利用检测模型得到的结果,与检测框有关   #####################
                        "box": [79.5, 79.0],    # 检测框的坐标 (box格式)
                        "conf": 0.95, # 检测框的置信度
                        "cls": 0, # 检测框的索引
                        "name": "target",  # 检测框的名称
                        "points": [[79.5, 79.0], [79.5, 79.0], [79.5, 79.0], [79.5, 79.0]], # 检测框的坐标( polygon 格式)
                        #################### 下面是利用分类模型得到的结果,与分类框有关   #####################
                        "top1name": "好",  # name即对应的分类名称
                        "top1conf": 0.95, # Confidence score
                        "top1": 0, # top1的索引
                        "top5name": ["好", "坏", "一般", "未知", "其他"], # top5的名称
                        "top5conf": [0.55, 0.1, 0.1, 0.1, 0.1], # top5的置信度
                        "top5": [0, 1, 2, 3, 4], # top5的索引
                        ##############原图片的信息#####################
                        “orig_shape”: (640, 640), # 原图片的大小, (宽, 高)
                    },
                    ....
                ]
            '''
            # 把结果画框出来
            font_style = ImageFont.truetype(cdata.FONT_SIMSUN, 20)
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            # 遍历每个框,画框
            for _idx, bbox in enumerate(results):
                points = bbox["points"]
                x1, y1 = points[0]
                x2, y2 = points[2]
                draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
                draw.text((x1 - 10, y1 - 10), str(bbox["name"]), font=font_style, fill="blue")
            img.save("output/aaa.png")
            ```
        """
        # Step 1: Perform object detection
        img = load_image(source)
        orig_shape = img.size
        det_results = self.det.detect(img)[0]
        results = []
        for _, item in enumerate(det_results):
            box: list = item["box"]
            x1, y1, x2, y2 = map(int, box)
            cropped_img = img.crop((x1, y1, x2, y2))
            cls_results = self.cls.classify(cropped_img)[0]
            d = {**item, **cls_results, "orig_shape": orig_shape}
            results.append(deepcopy(d))
        return results

    def similarity(
        self, img_path1: str | Path, img_path2: str | Path, threshold: float = 0.2
    ) -> float:
        """计算两个图片的相似度

        Args:
            img_path1 (str | Path): 第一张图片的路径
            img_path2 (str | Path): 第二张图片的路径
            threshold (float): 相似度阈值,默认值为 0.2

        Returns:
            float: 两张图片的相似度,范围在 [0.0, 1.0] 之间

        Example:
            ```python
            from cfun.yolo.detclsonnx import DetClsOnnx
            yolo = DetClsOnnx()
            img_path1 = "assets/image_cls_01.png"
            img_path2 = "assets/image_cls_02.png"
            sim = yolo.similarity(img_path1, img_path2, threshold=0.2)
            print(f"相似度: {sim}")
            ```
        """
        # 计算两个图片的 Jaccard 相似度
        sim = self.cls.similarity(img_path1, img_path2, threshold=threshold)
        return sim

    def draw_results(
        self,
        img: str | Path | Image.Image | np.ndarray,
        detections: list,
        save_path: str,
    ) -> None:
        """在图像上绘制检测结果

        Args:
            img (PIL.Image): Original image.
            detections (list): List of detected objects.
            save_path (str): Path to save the result image.

        Example:
            ```python
            from cfun.yolo.detclsonnx import DetClsOnnx
            yolo = DetClsOnnx()
            image_path = "assets/image_detect_01.png"
            results = yolo.predict(image_path)
            yolo.draw_results(img_path1, results, save_path="result1.png")
            ```
        """
        self.det.draw_results(img, detections, save_path)


if __name__ == "__main__":
    yolo = DetClsOnnx()
    image_path = "assets/image_detect_01.png"
    results = yolo.predict(image_path)
    import pprint

    print("检测结果：")
    pprint.pprint(results)

    # 计算两个图片的相似度
    img_path1 = "assets/image_cls_01.png"
    img_path2 = "assets/image_cls_02.png"
    sim = yolo.similarity(img_path1, img_path2, threshold=0.0)
    print(f"相似度: {sim}")

    # 把结果画框出来

    yolo.draw_results(
        image_path,
        results,
        save_path="output/aaa_onnx.png",
    )

    # style = ImageFont.truetype(cdata.FONT_SIMSUN, 20)  # 设置字体和大小
    # img = Image.open(image_path)  # 打开图片
    # draw = ImageDraw.Draw(img)  # 创建一个可以在图片上绘制的对象(相当于画布)
    # # 遍历每个框,画框
    # for _idx, bbox in enumerate(results):
    #     x1, y1, x2, y2 = map(int, bbox["box"])

    #     # 画框, outline参数用来设置矩形边框的颜色
    #     draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
    #     # 写字 (偏移一定的距离)
    #     draw.text((x1 - 10, y1 - 10), str(bbox["top1name"]), font=style, fill="blue")
    # # 保存图片
    # img.save("output/aaa.png")

    # img_path1 = "assets/image_cls_01.png"
    # img_path2 = "assets/image_cls_02.png"
    # sim = yolo.similarity(img_path1, img_path2, threshold=0.2)
    # print(f"相似度: {sim}")
