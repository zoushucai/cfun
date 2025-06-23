from cfun.yolo.convert import box_to_polygon, polygon_to_box


def test_box():
    box = {"box": [10, 20, 30, 40], "cls": "cat", "conf": 0.9}
    polygon = box_to_polygon(box)
    print(polygon)
    out1 = {
        "points": [[10, 20], [30, 20], [30, 40], [10, 40]],
        "cls": "cat",
        "conf": 0.9,
    }
    assert polygon == out1, f"Expected {out1}, but got {polygon}"
    box_list = [
        {"box": [10, 20, 30, 40], "cls": "cat", "conf": 0.9},
        {"box": [50, 60, 70, 80], "cls": "dog", "conf": 0.8},
    ]
    polygons = box_to_polygon(box_list)
    print(polygons)
    out1 = [
        {"points": [[10, 20], [30, 20], [30, 40], [10, 40]], "cls": "cat", "conf": 0.9},
        {"points": [[50, 60], [70, 60], [70, 80], [50, 80]], "cls": "dog", "conf": 0.8},
    ]
    assert polygons == out1, f"Expected {out1}, but got {polygons}"
    # 注意: 这里的 box 表示是一个字典,包含了 box 的坐标、类别和置信度


def test_polygon():
    polygon = {
        "points": [[10, 20], [30, 20], [30, 40], [10, 40]],
        "name": "cat",
        "confidence": 0.9,
    }
    box = polygon_to_box(polygon)
    print(box)
    # 输出: {"box": [10, 20, 30, 40], "name": "cat", "confidence": 0.9}

    polygon_list = [
        {
            "points": [[10, 20], [30, 20], [30, 40], [10, 40]],
            "name": "cat",
            "confidence": 0.9,
        },
        {
            "points": [[50, 60], [70, 60], [70, 80], [50, 80]],
            "name": "dog",
            "confidence": 0.8,
        },
    ]
    boxes = polygon_to_box(polygon_list)
    print(boxes)
    out1 = [
        {"box": [10, 20, 30, 40], "name": "cat", "confidence": 0.9},
        {"box": [50, 60, 70, 80], "name": "dog", "confidence": 0.8},
    ]
    assert boxes == out1, f"Expected {out1}, but got {boxes}"


if __name__ == "__main__":
    test_box()
    test_polygon()
