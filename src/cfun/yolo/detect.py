from pathlib import Path
from typing import Any, Optional, Sequence, Union

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

from ..font import get_chinese_font_path_random
from .base import extract_info_from_onnx, load_image


class Detector:
    """yolo检测器

    Attributes:
        session (onnxruntime.InferenceSession): ONNX 推理会话
        imgsz (tuple[int, int]): 模型输入图像尺寸
    """

    def __init__(
        self,
        model: str | Path,
        names: Optional[dict[int, str]] = None,
        imgsz: int | tuple[int, int] | list[int] | None = (640, 640),
        providers: Optional[list] = None,
    ):
        """初始化

        Args:
            model (str | Path): 模型路径
            names (dict[int, str]): 类别字典,如 {0: "cat", 1: "dog", ...}, 如果不传入,则尝试从onnx模型中提取
            imgsz (tuple[int, int], optional): 模型输入图像尺寸 (W, H). Defaults to (640, 640).
            providers (Optional[list], optional): ONNX 推理后端. Defaults to None.

        """
        if providers is None:
            providers = ["CPUExecutionProvider"]
        if isinstance(model, str):
            model = Path(model)
        self.session = ort.InferenceSession(model, providers=providers)
        self._input_name = self.session.get_inputs()[0].name

        self.names = names if names else extract_info_from_onnx(model, "names")
        if not self.names:
            raise ValueError("names must be provided or extracted from ONNX metadata")
        if not isinstance(self.names, dict):
            raise TypeError("names must be a dictionary")

        if isinstance(imgsz, int):
            imgsz = (imgsz, imgsz)
        elif isinstance(imgsz, list):
            if len(imgsz) == 1:
                imgsz = (imgsz[0], imgsz[0])
            elif len(imgsz) == 2:
                imgsz = (imgsz[0], imgsz[1])
            else:
                raise ValueError("imgsz must be int, tuple or list of 1 or 2 integers")
        elif isinstance(imgsz, tuple):
            if len(imgsz) == 1:
                imgsz = (imgsz[0], imgsz[0])
            elif len(imgsz) == 2:
                imgsz = (imgsz[0], imgsz[1])
            else:
                raise ValueError("imgsz tuple must have 1 or 2 elements")
        else:
            raise ValueError("imgsz must be int, tuple or list of 1 or 2 integers")

        self.imgsz = imgsz

        self.model = model  # 后续未使用,暂存

    def _letterbox(
        self,
        img: Image.Image,
        new_shape: tuple[int, int] = (640, 640),
        color: tuple[int, int, int] = (114, 114, 114),
        scaleup: bool = True,
    ) -> tuple[Image.Image, float, tuple[float, float]]:
        """重新调整图像大小,保持纵横比并填充空白区域

        Args:
            img (PIL.Image): Input image.
            new_shape (tuple): New shape for the image (width, height).
            color (tuple): Color for the padding (R, G, B).
            scaleup (bool): Whether to scale up the image if it's smaller than new_shape.

        Returns:
            tuple: Tuple containing the resized image, the resize ratio, and the padding (dw, dh).

        """
        shape = img.size  # (width, height)
        r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
        if not scaleup:
            r = min(r, 1.0)

        new_unpad = (int(round(shape[0] * r)), int(round(shape[1] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        img = img.resize(new_unpad, Image.Resampling.BILINEAR)
        new_img = Image.new("RGB", new_shape, color)
        new_img.paste(img, (int(dw), int(dh)))
        return new_img, r, (dw, dh)

    def _preprocess(self, source: Union[str, Path, Image.Image, np.ndarray]):
        """预处理单个图像"""
        img0 = load_image(source)
        img, ratio, (dw, dh) = self._letterbox(img0, self.imgsz)
        img = np.array(img).astype(np.float32) / 255.0  # HWC, RGB
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # NCHW
        return img, img0, ratio, dw, dh

    def _infer(self, img):
        return self.session.run(None, {self._input_name: img})[0]

    def _detect_single(
        self,
        source: Union[str, Path, Image.Image, np.ndarray],
        conf_thres: float = 0.65,
    ) -> list[dict[str, Union[int, float, list[float]]]]:
        """处理单张图像,返回检测结果"""

        img_tensor, img0, ratio, dw, dh = self._preprocess(source)
        preds = self._infer(img_tensor)[0]

        h0, w0 = img0.size[1], img0.size[0]
        detections = []

        for det in preds:
            x1, y1, x2, y2, conf, cls0 = det
            if conf < conf_thres:
                continue
            x1 = np.clip((x1 - dw) / ratio, 0, w0 - 1)
            y1 = np.clip((y1 - dh) / ratio, 0, h0 - 1)
            x2 = np.clip((x2 - dw) / ratio, 0, w0 - 1)
            y2 = np.clip((y2 - dh) / ratio, 0, h0 - 1)

            x1, y1, x2, y2, cls0 = int(x1), int(y1), int(x2), int(y2), int(cls0)
            detections.append(
                {
                    "cls": cls0,
                    "name": self.names[cls0],
                    "conf": round(float(conf), 2),
                    "box": [x1, y1, x2, y2],
                    "points": [
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2],
                    ],
                }
            )
        if not detections:
            raise ValueError("No detections found in the image")
        return detections

    def detect(
        self, source: Union[str, Path, Image.Image, np.ndarray, Sequence]
    ) -> list[list[dict[str, Any]]]:
        """检测图像中的物体,支持多种输入类型和批量检测。

        Args:
            source: 单张图像或图像列表。支持路径、PIL.Image 或 cv2 读取的图像。

        Returns:
            list[dict]: 检测结果列表,每个元素是一个字典,包含检测到的物体信息。

                - `box` (list[float]): 检测框的坐标,box 格式。
                - `conf` (float): 检测框的置信度。
                - `cls` (int): 检测框的类别索引。
                - `name` (str): 检测框的名称。
                - `points` (list[list[float]]): 检测框的多边形坐标。

        Example:
            ```python
            from cfundata import cdata

            det = Detector(cdata.DX_DET_ONNX)

            img_path1 = "assets/image_detect_01.png"
            img_path2 = "assets/image_detect_02.png"
            img_path3 = "assets/image_detect_03.png"


            # 单图像检测
            result1 = det.detect(img_path1)[0]
            print("-----单图像检测1-----")
            print(result1)

            result2 = det.detect(img_path2)[0]
            print("-----单图像检测2-----")
            print(result2)

            result3 = det.detect(img_path3)[0]
            print("-----单图像检测3-----")
            print(result3)

            # 批量检测
            result_all = det.detect([img_path1, img_path2, img_path3])
            print("-----批量检测-----")
            print(result_all)
            # 传入 cv2 读取的图像
            img_cv2 = cv2.imread(img_path1)
            result_cv2 = det.detect(img_cv2)[0]
            print("-----cv2读取的图像-----")
            print(result_cv2)
            # 传入 PIL.Image
            img_pil = Image.open(img_path1)
            result_pil = det.detect(img_pil)[0]
            print("-----PIL读取的图像-----")
            print(result_pil)

            # 混合传入
            result_mixed = det.detect([img_path1, img_cv2, img_pil])
            print("-----混合传入-----")
            print(result_mixed)

            ```
        """
        # 单图像转列表
        if not isinstance(source, Sequence) or isinstance(
            source, (str, Path, Image.Image, np.ndarray)
        ):
            img_list = [source]
        else:
            img_list = source
        results = [self._detect_single(img) for img in img_list]
        if not results:
            raise ValueError("No detections found in the image(s)")
        # 过滤掉空列表
        results = [result for result in results if result]
        return results

    def draw_results(
        self,
        img: str | Path | Image.Image | np.ndarray,
        detections: list[dict[Any, Any]] | dict,
        save_path: Union[str, Path],
    ) -> None:
        """在图像上绘制检测结果

        Args:
            img (PIL.Image): Original image.
            detections (list): List of detected objects.
            save_path (str): Path to save the result image.

        Example:
            ```python
            from cfundata import cdata
            det = Detector(cdata.DX_DET_ONNX)
            img_path1 = "assets/image_detect_01.png"
            result1 = det.detect(img_path1)
            det.draw_results(img_path1, result1[0], save_path="result1.png")
            ```
        """
        if isinstance(detections, dict):
            detections = [detections]
        if not isinstance(detections, list):
            raise TypeError("detections must be a list or a dict")
        if not detections:
            raise ValueError("detections list is empty")

        img = load_image(img)
        assert isinstance(img, Image.Image), "img must be a PIL.Image"
        img = img.convert("RGB")
        fontpath = get_chinese_font_path_random()
        if fontpath is None:
            raise ValueError(
                "Font path returned None. Please check your font configuration."
            )
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
        img.save(str(save_path))


if __name__ == "__main__":
    from cfundata import cdata

    det = Detector(cdata.DX_DET_ONNX)

    img_path1 = "assets/image_detect_01.png"
    img_path2 = "assets/image_detect_02.png"
    img_path3 = "assets/image_detect_03.png"

    # 单图像检测
    result1 = det.detect(img_path1)[0]
    print("-----单图像检测1-----")
    print(result1)

    result2 = det.detect(img_path2)[0]
    print("-----单图像检测2-----")
    print(result2)

    result3 = det.detect(img_path3)[0]
    print("-----单图像检测3-----")
    print(result3)

    # 批量检测
    result_all = det.detect([img_path1, img_path2, img_path3])
    print("-----批量检测-----")
    print(result_all)
    # 传入 cv2 读取的图像
    img_cv2 = cv2.imread(img_path1)
    result_cv2 = det.detect(img_cv2)[0]
    print("-----cv2读取的图像-----")
    print(result_cv2)
    # 传入 PIL.Image
    img_pil = Image.open(img_path1)
    result_pil = det.detect(img_pil)[0]
    print("-----PIL读取的图像-----")
    print(result_pil)

    # 混合传入
    result_mixed = det.detect([img_path1, img_cv2, img_pil])[0]
    print("-----混合传入-----")
    print(result_mixed)

    # 绘制检测结果
    det.draw_results(img_path1, [result1[0]], save_path="output/detect1.png")
    det.draw_results(img_path2, [result2[0]], save_path="output/detect2.png")
    det.draw_results(img_path3, [result3[0]], save_path="output/detect3.png")
    det.draw_results(img_cv2, [result_cv2[0]], save_path="output/detect_cv2.png")
    det.draw_results(img_pil, [result_pil[0]], save_path="output/detect_pil.png")

    print("-----绘制检测结果完成-----")
    # 画框
    det.draw_results(
        img_path1,
        [result1[0]],
        save_path="output/aaa_2.png",
    )
