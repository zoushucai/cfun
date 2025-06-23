from pathlib import Path

from cfun.yolo.detclsonnx import DetClsOnnx


def test_detclsonnx():
    image_path = Path(__file__).parent / "images" / "image_detect_01.png"
    yolo = DetClsOnnx()
    results = yolo.predict(image_path)
    # Output the result
    print(results)
    # 把结果画框出来
    yolo.draw_results(
        image_path,
        results,
        save_path="output/aaa_testonnx.png",
    )


if __name__ == "__main__":
    test_detclsonnx()
