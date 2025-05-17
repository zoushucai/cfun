import cv2
from cfundata import cdata
from PIL import Image

from cfun.yolo.classify import Classifier
from cfun.yolo.detect import Detector


def test_detect():
    det = Detector(model=cdata.DX_DET_ONNX)

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

    # 绘制检测结果
    det.draw_results(img_path1, result1[0], save_path="output/result1.png")
    det.draw_results(img_path2, result2[0], save_path="output/result2.png")
    det.draw_results(img_path3, result3[0], save_path="output/result3.png")
    det.draw_results(img_cv2, result_cv2[0], save_path="output/result_cv2.png")
    det.draw_results(img_pil, result_pil[0], save_path="output/result_pil.png")
    print("-----绘制检测结果完成-----")


def test_classify():
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


if __name__ == "__main__":
    test_detect()
    test_classify()
