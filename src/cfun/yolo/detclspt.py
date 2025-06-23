"""
主要是根据 yolo11的目标检测和分类模型,对图片进行检测和分类,返回一个list[dict],每个元素是一个字典,包含名称和坐标, 依赖 ultralytics

1. 首先通过检测模型,得到检测框的坐标
2. 然后根据检测框的坐标,裁剪出图片
3. 然后通过分类模型,对裁剪出来的图片进行分类
4. 最后将分类结果和坐标一起返回, 返回一个list[dict]


"""

from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO  # 可以对实验进行精细化设置
from ultralytics.engine.results import Results

from ..font import get_chinese_font_path_random
from .base import fillter_top5, jaccard_similarity, load_image


class DetClsPt:
    """YOLO模型类,用于目标检测和分类, 传入的是pt模型, 只适用于一张图上同时有检测和分类的情况

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
            det_model (one | str | Path): 检测模型路径,默认为None,使用默认模型
            cls_model (one | str | Path): 分类模型路径,默认为None,使用默认模型
            det_imgsz (int | tuple[int, int] | list[int,int]): 检测模型的输入大小
            cls_imgsz (int | tuple[int, int] | list[int, int]): 分类模型的输入大小

        """
        if det_model is None and cls_model is None:
            from cfundata import cdata

            det_model = cdata.DX_DET_PT  # 内部检测模型
            cls_model = cdata.DX_CLS_PT  # 内部分类模型
        # 加载模型
        self.det = YOLO(self._resolve_model_path(det_model, "Detection"))

        self.cls = YOLO(self._resolve_model_path(cls_model, "Classification"))
        self.det_imgsz = det_imgsz  # 检测模型的输入大小
        self.cls_imgsz = cls_imgsz

    @staticmethod
    def _resolve_model_path(path, name):
        assert path, f"{name} model path is None"
        path = str(path) if isinstance(path, Path) else path
        path_obj = Path(path)

        assert path_obj.exists(), f"{name} model not found: {path}"
        assert path_obj.suffix == ".pt", f"{name} model must be a .pt file"

        return path

    def _load_image(
        self, source: Union[str, Path, Image.Image, np.ndarray]
    ) -> Image.Image:
        """加载图像,支持路径、PIL.Image 或 OpenCV 图像（ndarray）

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
            if source.ndim == 3 and source.shape[2] == 3:
                img0 = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
            else:
                raise ValueError("Unsupported ndarray format: must be HxWx3 BGR image")
        else:
            raise TypeError(
                "source must be a file path, PIL.Image, or cv2 image (ndarray)"
            )

        return img0

    def predict(self, source: Union[str, Path, Image.Image, np.ndarray]) -> list[dict]:
        """根据输入的图片路径进行检测和分类,并返回结果 (单个图片)

        Args:
            source (Union[str, Path, Image.Image, np.ndarray]): 输入的图片路径或图片对象

        Returns:
            list[dict]: 检测和分类结果,包含名称和坐标等

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
                - `orig_shape` (tuple[int, int]): 原始图片的大小。(宽, 高)

        Example:
            ```python
            from cfun.yolo.detclspt import DetClsPt
            yolo = DetClsPt()
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
                        “orig_shape”: (640, 640), # 原图片的大小 (宽, 高)
                    },
                    ....
                ]
            '''
            ```
        """
        # Step 1: Perform object detection
        det_results = self.det.predict(source, imgsz=self.det_imgsz, verbose=False)

        names = det_results[0].names  # All class names
        orig_shape = det_results[0].orig_shape  # Original image shape
        # Get bounding box coordinates
        if len(det_results) == 0:
            return []
        boxes = det_results[0].boxes
        if boxes is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy().tolist()  # type: ignore
        cls = boxes.cls.cpu().numpy().tolist()  # type: ignore
        conf = boxes.conf.cpu().numpy().tolist()  # type: ignore
        # Step 2: Load image for cropping
        img = self._load_image(source)

        results = []
        # Step 3: For each detected box, crop the image and classify
        for box, cl, co in zip(xyxy, cls, conf, strict=False):
            # print(item)
            # print("--" * 20)
            # print(item.cls)
            x1, y1, x2, y2 = map(int, box)  # Convert to integers
            # Crop the image using the bounding box
            cropped_img = img.crop((x1, y1, x2, y2))

            # Step 4: Perform classification on the cropped image
            cls_results = self.cls.predict(
                cropped_img, imgsz=self.cls_imgsz, verbose=False
            )

            # Step 5: Process classification results
            assert len(cls_results) == 1, (
                "Classification result should be a single output."
            )
            clsdict = self._extract_classify_result(cls_results)
            # Step 6: Store the result

            detdict = {
                "box": [x1, y1, x2, y2],  # Detection box coordinates
                "conf": round(co, 4),  # Detection box confidence
                "cls": int(cl),
                "name": names[int(cl)],
                "points": [
                    [x1, y1],
                    [x1, y2],
                    [x2, y2],
                    [x2, y1],
                ],  # Detection box coordinates in polygon format
                "orig_shape": (
                    orig_shape[1],
                    orig_shape[0],
                ),  # Original image size (width, height)
            }

            results.append({**detdict, **clsdict})

        return results

    @staticmethod
    def _extract_classify_result(cls_results: list[dict] | List[Results]) -> dict:
        """提取分类结果

        Args:
            cls_results (dict): 检测和分类结果, yolo.predict() 的返回值

        Returns:
            dict: 提取后的分类结果
        """
        r = cls_results[0]
        if r is None:
            return {
                "top1name": None,
                "top1conf": None,
                "top1": None,
                "top5name": None,
                "top5conf": None,
                "top5": None,
            }

        all_names = r.names  # type: ignore # All class names
        top1 = r.probs.top1  # type: ignore # Index of the top prediction
        top1name = all_names[top1]  # Top prediction name
        top1conf = r.probs.top1conf.item()  # type: ignore # Top prediction confidence

        top5 = r.probs.top5  # type: ignore # Indices of the top 5 predictions
        top5name = [all_names[i] for i in top5]
        top5conf = r.probs.top5conf.tolist()  # type: ignore
        return {
            "top1name": top1name,  # Name of the top prediction
            "top1conf": round(top1conf, 4),
            "top1": top1,  # Index of the top prediction
            "top5name": top5name,  # Names of the top 5 predictions
            "top5conf": [round(i, 4) for i in top5conf],
            "top5": top5,
        }

    # 计算两个图片的相似度
    def similarity(
        self, img_path1: str | Path, img_path2: str | Path, threshold: float = 0.2
    ) -> float:
        """计算两个图片的相似度, 采用 Jaccard 相似度

        原理: 首先计算两个图片的 top5 类别,类别的概率应该大于阈值,然后计算这两个集合的 Jaccard 相似度。

        Args:
            img_path1 (str | Path): 图片路径1
            img_path2 (str | Path): 图片路径2

        Returns:
            float: 相似度

        Example:
            ````python
            from cfun.yolo.detclspt import DetClsPt
            yolo = DetClsPt()
            sim = yolo.similarity("path/to/image1.jpg", "path/to/image2.jpg")
            print(sim)
            ````
        """

        result1 = self.cls.predict(img_path1)
        result2 = self.cls.predict(img_path2)

        data1 = self._extract_classify_result(result1)
        data2 = self._extract_classify_result(result2)
        # 对前五个的概率 进行筛选,至少大于 threshold
        # 根据 top5conf 过滤
        result1 = fillter_top5(data1, threshold)
        result2 = fillter_top5(data2, threshold)
        top5_1 = result1["top5"]
        top5_2 = result2["top5"]

        # 计算相似度
        sim = jaccard_similarity(top5_1, top5_2)
        return sim  # 保留两位小数

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
            from cfun.yolo.detclspt import DetClsPt
            image_path = "assets/image_detect_01.png"
            yolo = DetClsPt()
            results = yolo.predict(image_path)
            print(results)
            # 把结果画框出来
            yolo.draw_results(
                image_path,
                results,
                save_path="output/aaa_testpt.png",
            )
            ```
        """
        img = load_image(img)
        assert isinstance(img, Image.Image), "img must be a PIL.Image"
        img = img.convert("RGB")
        fontpath = get_chinese_font_path_random()
        if fontpath is None:
            raise ValueError("Font path cannot be None")
        style = ImageFont.truetype(str(fontpath), 20)
        draw = ImageDraw.Draw(img)  # 创建一个可以在图片上绘制的对象(相当于画布)
        for det in detections:
            box = det["box"]
            conf = det["conf"]
            x1, y1, x2, y2 = box
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            name = det["top1name"] if "top1name" in det else det["name"]
            draw.text((x1 - 10, y1 - 10), f"{name} {conf:.2f}", font=style, fill="blue")
        #
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)


if __name__ == "__main__":
    yolo = DetClsPt()
    image_path = "assets/image_detect_01.png"
    results = yolo.predict(image_path)
    print(results)

    # 计算两个图片的相似度
    img_path1 = "assets/image_cls_01.png"
    img_path2 = "assets/image_cls_02.png"
    sim = yolo.similarity(img_path1, img_path2, threshold=0.0)
    print(f"相似度: {sim}")

    # 把结果画框出来
    yolo.draw_results(
        image_path,
        results,
        save_path="output/aaa_pt.png",
    )

    # font_style = ImageFont.truetype(cdata.FONT_SIMSUN, 20)  # 设置字体和大小
    # img = Image.open(image_path)  # 打开图片
    # draw = ImageDraw.Draw(img)  # 创建一个可以在图片上绘制的对象(相当于画布)
    # # 遍历每个框,画框
    # for _idx, bbox in enumerate(results):
    #     points = bbox["points"]
    #     x1, y1 = points[0]
    #     x2, y2 = points[2]
    #     # 画框, outline参数用来设置矩形边框的颜色
    #     draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
    #     # 写字 (偏移一定的距离)
    #     draw.text((x1 - 10, y1 - 10), str(bbox["name"]), font=font_style, fill="blue")
    # # 保存图片
    # img.save("output/aaa.png")
