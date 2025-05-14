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
    # font_path = str(cdata.FONT_SIMSUN)
    # print(f"font_path: {font_path}")
    # style = ImageFont.truetype(cdata.FONT_SIMSUN, 20)  # 设置字体和大小
    # img = Image.open(image_path)  # 打开图片
    # draw = ImageDraw.Draw(img)  # 创建一个可以在图片上绘制的对象(相当于画布)
    # # 遍历每个框，画框
    # for _idx, bbox in enumerate(results):
    #     x1, y1, x2, y2 = map(int, bbox["box"])
    #     # 画框, outline参数用来设置矩形边框的颜色
    #     draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
    #     # 写字 (偏移一定的距离)
    #     draw.text((x1 - 10, y1 - 10), str(bbox["top1name"]), font=style, fill="blue")
    # # 保存图片
    # Path("output").mkdir(parents=True, exist_ok=True)  # 创建输出目录
    # img.save("output/aaa_test.png")


if __name__ == "__main__":
    test_detclsonnx()
