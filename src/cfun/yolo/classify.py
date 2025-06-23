from pathlib import Path, PosixPath, WindowsPath
from typing import Optional, Sequence, Union

import numpy as np
import onnxruntime as ort
from PIL import Image

from .base import extract_info_from_onnx, fillter_top5, jaccard_similarity, load_image


class Classifier:
    """yolo 分类器(采用 PIL 库处理图像)

    Attributes:
        session (onnxruntime.InferenceSession): ONNX 推理会话
        imgsz (tuple[int, int]): 模型输入图像尺寸
    """

    def __init__(
        self,
        model: str | Path | PosixPath | WindowsPath,
        names: Optional[dict[int, str]] = None,
        imgsz: int | tuple[int, int] | list[int] | None = (64, 64),
        providers: Optional[list] = None,
    ):
        """初始化分类器

        Args:
            model (str | Path): 模型路径
            names (dict[int, str]): 类别字典,如 {0: "cat", 1: "dog", ...}, 如果不传入,则尝试从onnx模型中提取
            imgsz (tuple[int, int], optional): 模型输入图像尺寸 (W, H). Defaults to (64, 64).
            providers (Optional[list], optional): ONNX 推理后端. Defaults to None.
        """
        if isinstance(model, str):
            model = Path(model)

        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model, providers=providers)
        self._input_name = self.session.get_inputs()[0].name

        self.names = names if names else extract_info_from_onnx(model, "names")
        if not self.names:
            raise ValueError("names must be provided or extracted from ONNX metadata")
        if not isinstance(self.names, dict):
            raise TypeError("names must be a dictionary")

        if isinstance(imgsz, int):
            imgsz = (imgsz, imgsz)
        if isinstance(imgsz, list):
            if len(imgsz) == 1:
                imgsz = (imgsz[0], imgsz[0])
            elif len(imgsz) == 2:
                imgsz = (imgsz[0], imgsz[1])
            else:
                raise ValueError(
                    f"imgsz must be a tuple or list of two integers, but got {imgsz!r}"
                )
        self.imgsz = (64, 64) if imgsz else extract_info_from_onnx(model, "imgsz")
        self.model = model  # 后续未使用,暂存

    def _preprocess(
        self, source: Union[str, Path, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """预处理单个图像

        Args:
            img_path (Union[str, Path, Image.Image, np.ndarray]): 图像路径或图像对象

        Returns:
            np.ndarray: 预处理后的图像（NCHW 格式）

        """
        img = load_image(source)
        # 调整大小,后面的是一种插值算法
        # 确保 self.imgsz 是 (W, H) 的元组或列表
        if not (
            isinstance(self.imgsz, (tuple, list))
            and len(self.imgsz) == 2
            and all(isinstance(x, int) for x in self.imgsz)
        ):
            raise ValueError(
                f"imgsz must be a tuple or list of two integers, but got {self.imgsz!r}"
            )
        img = img.resize(self.imgsz, Image.Resampling.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0  # 转为 float32 并归一化
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)  # 添加 batch 维度 → NCHW
        # ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组,使得运行速度更快。
        img = np.ascontiguousarray(img)
        return img

    def _infer(self, img):
        """推理"""
        return self.session.run(None, {self._input_name: img})[0]

    def _classify_single(
        self, source: Union[str, Path, Image.Image, np.ndarray]
    ) -> dict:
        """根据输入的图像进行分类

        Args:
            source (Union[str, Path, Image.Image, np.ndarray]): 图像路径或图像对象

        Returns:
            dict: 分类结果,包含 top1, top1name, top1conf, top5, top5name, top5conf

                - `top1name` (str): 分类结果的名称（Top-1）。
                - `top1conf` (float): 分类结果的置信度（Top-1）。
                - `top1` (int): 分类结果的索引（Top-1）。
                - `top5name` (list[str]): 前5个分类结果的名称。
                - `top5conf` (list[float]): 前5个分类结果的置信度。
                - `top5` (list[int]): 前5个分类结果的索引。


        Example:
            ````python
            from cfun.yolo.classify import Classifier
            # 假设你有一个名为 "model.onnx" 的 ONNX 模型文件
            # 并且你有一个类别字典 names, 例如 {0: "cat", 1: "dog"}
            classifier = Classifier(model="model.onnx", names={0: "cat", 1: "dog"})
            result = classifier.classify("path/to/image.jpg")
            print(result)  # 输出预测结果
            ````

        """
        img = self._preprocess(source)
        assert isinstance(img, np.ndarray), "Image must be a numpy array"
        outputs = self._infer(img)
        probs = outputs[0]
        if probs.ndim == 2:
            probs = probs[0]

        # 处理概率
        # 获取 top-5 索引（按概率从高到低）
        top5_indices = np.argsort(probs)[-5:][::-1].astype(int).tolist()

        # 构建 top-5 概率（保留 4 位小数）和名称
        top5_conf = [round(float(probs[i]), 4) for i in top5_indices]
        # 如果不对应,则报错,这是希望看见的

        top5_names = [self.names[i] for i in top5_indices]

        # Top-1 就是 top5 的第一个
        data = {
            "top1": top5_indices[0],  # 预测类别 ID
            "top1name": top5_names[0],  # 预测类别名称
            "top1conf": top5_conf[0],  # 预测置信度
            "top5": top5_indices,  # 前五个类别 ID
            "top5name": top5_names,  # 前五个类别名称
            "top5conf": top5_conf,  # 前五个置信度
        }
        return data

    def classify(
        self,
        source: Union[str, Path, Image.Image, np.ndarray, Sequence],
    ) -> list[dict]:
        """图像分类函数,支持单个或多个图像,图像可以是路径、PIL.Image 或 OpenCV 图像。

        Args:
            source: 图像路径、PIL.Image、OpenCV 图像或它们的列表。

        Returns:
            list[dict]: 每张图像的分类结果（字典）。

                - `top1name` (str): 分类结果的名称（Top-1）。
                - `top1conf` (float): 分类结果的置信度（Top-1）。
                - `top1` (int): 分类结果的索引（Top-1）。
                - `top5name` (list[str]): 前5个分类结果的名称。
                - `top5conf` (list[float]): 前5个分类结果的置信度。
                - `top5` (list[int]): 前5个分类结果的索引。
        Example:
            ````python
            from cfun.yolo.classify import Classifier
            # 假设你有一个名为 "model.onnx" 的 ONNX 模型文件
            # 并且你有一个类别字典 names, 例如 {0: "cat", 1: "dog"}
            classifier = Classifier(model="model.onnx", names={0: "cat", 1: "dog"})
            result = classifier.classify("path/to/image.jpg")
            print(result)


            ````
        """
        # 单图像变列表
        if not isinstance(source, Sequence) or isinstance(
            source, (str, Path, Image.Image, np.ndarray)
        ):
            img_list = [source]
        else:
            img_list = list(source)

        results = [self._classify_single(img) for img in img_list]

        return results

    # 计算两个图片的相似度
    def similarity(
        self,
        img_path1: str | Path | Image.Image | np.ndarray,
        img_path2: str | Path | Image.Image | np.ndarray,
        threshold: float = 0.2,
    ) -> float:
        """计算两个图片的相似度, 采用 Jaccard 相似度

        原理: 首先计算两个图片的 top5 类别,类别的概率应该大于阈值,然后计算这两个集合的 Jaccard 相似度。

        Args:
            img_path1 (str | Path | Image.Image | np.ndarray): 第一张图片的路径或图像对象
            img_path2 (str | Path | Image.Image | np.ndarray): 第二张图片的路径或图像对象
            threshold (float, optional): 置信度阈值,默认值为 0.2

        Returns:
            float: 相似度

        Example:
            ````python
            from cfun.yolo.classify import Classifier
            # 假设你有一个名为 "model.onnx" 的 ONNX 模型文件
            classifier = Classifier(model="model.onnx")
            sim = classifier.similarity("path/to/image1.jpg", "path/to/image2.jpg")
            print(sim)
            # 输出相似度
            ````
        """

        result = self.classify([img_path1, img_path2])
        # 对前五个的概率 进行筛选,至少大于 threshold
        # 根据 top5conf 过滤
        result1 = fillter_top5(result[0], threshold)
        result2 = fillter_top5(result[1], threshold)
        top5_1 = result1["top5"]
        top5_2 = result2["top5"]

        # 计算相似度
        sim = jaccard_similarity(top5_1, top5_2)
        return sim  # 保留两位小数


if __name__ == "__main__":
    # from all_name import names

    import cv2
    from cfundata import cdata

    # 加载自己的模型文件
    classifier = Classifier(model=cdata.DX_CLS_ONNX)

    img_path1 = "assets/image_cls_01.png"
    img_path2 = "assets/image_cls_02.png"
    img_path3 = "assets/image_cls_03.png"

    # 单图像检测
    result1 = classifier.classify(img_path1)
    print("-----单图像检测1-----")
    print(result1)
    result2 = classifier.classify(img_path2)
    print("-----单图像检测2-----")
    print(result2)
    result3 = classifier.classify(img_path3)
    print("-----单图像检测3-----")
    print(result3)

    # 批量检测
    result_all = classifier.classify([img_path1, img_path2, img_path3])
    print("-----批量检测-----")
    print(result_all)
    # 传入 cv2 读取的图像
    img_cv2 = cv2.imread(img_path1)
    result_cv2 = classifier.classify(img_cv2)
    print("-----cv2读取的图像-----")
    print(result_cv2)
    # 传入 PIL.Image
    img_pil = Image.open(img_path1)
    result_pil = classifier.classify(img_pil)
    print("-----PIL读取的图像-----")
    print(result_pil)
    # 混合传入
    result_mixed = classifier.classify([img_path1, img_cv2, img_pil])
    print("-----混合传入-----")
    print(result_mixed)
    # 计算相似度
    sim = classifier.similarity(img_path1, img_path2, threshold=0.0)
    print("-----相似度-----")
    print(sim)

    # 计算相似度
    sim = classifier.similarity(img_path1, img_path3, threshold=0.0)
    print("-----相似度-----")
    print(sim)

    # 计算相似度--混合
    sim = classifier.similarity(
        img_cv2, img_path3, threshold=0.0
    )  # 图像1的cv2 和 图像3的路径
    print("-----相似度-----")
    print(sim)

    # 计算相似度--混合
    sim = classifier.similarity(
        img_path3, img_pil, threshold=0.0
    )  # 图像3的路径 和 图像1的PIL
    print("-----相似度-----")
    print(sim)
