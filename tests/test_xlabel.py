from cfun.yolo.xlabel import XLabel


def test_xlabel():
    image_path = "tests/images/image_detect_01.png"

    ### 矩形的例子
    data = [
        {
            "name": "char1",
            "points": [[100, 200], [200, 200], [200, 300], [100, 300]],
            "confidence": 0.9,
        },
        {
            "name": "char2",
            "points": [[300, 400], [400, 400], [400, 500], [300, 500]],
            "confidence": 0.8,
        },
    ]

    xl = XLabel(
        image_path,
        data,
        datakey="points",  # 这个key必须在data中存在
        platform="yolo",
        fixedtimestamp=True,
        namereplace={"name": "description", "confidence": "score"},
    )
    # 建议json的名字和图片的名字一致（这里是为了测试）
    xl.save_template("output/template1.json")  # 输出json文件

    #### 旋转矩形的例子
    image_path = "tests/images/image_detect_01.png"
    data = [
        {
            "name": "char1",
            "points": [[100, 200], [200, 200], [200, 300], [100, 300]],
            "confidence": 0.9,
            "angle": 45,  # 旋转角度
        },
        {
            "name": "char2",
            "points": [[300, 400], [400, 400], [400, 500], [300, 500]],
            "confidence": 0.8,
            "angle": 25,
        },
    ]
    namereplace = {
        "name": "description",
        "angle": "direction",  # 这里需要添加一个映射值为 direction的映射值
        "confidence": "score",
    }
    xl = XLabel(
        image_path,
        data,
        datakey="points",  # 这个key必须在data中存在
        platform="yolo",
        fixedtimestamp=True,
        shape_type="rotation",
        namereplace=namereplace,
    )
    xl.save_template("output/template2.json")


if __name__ == "__main__":
    test_xlabel()
    print("test_xlabel passed")
