from cfun.yzm.chaojiying import Chaojiying_Client


def test_chaojiying():
    # pass
    chaojiying = Chaojiying_Client("****", "****", "***")
    assert chaojiying is not None, "Chaojiying_Client 实例化失败"
    # image_path = Path(__file__).parent / "images" / "image_detect_01.png"
    # result = chaojiying.get(image_path, 9800)


if __name__ == "__main__":
    test_chaojiying()
