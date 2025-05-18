from pathlib import Path

from cfun.yolo.detclspt import DetClsPt


def test_detclspt():
    image_path = Path(__file__).parent / "images" / "image_detect_01.png"
    yolo = DetClsPt()
    results = yolo.predict(image_path)
    print(results)
    # 把结果画框出来
    yolo.draw_results(
        image_path,
        results,
        save_path="output/aaa_testpt.png",
    )


if __name__ == "__main__":
    test_detclspt()
