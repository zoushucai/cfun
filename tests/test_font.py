from pathlib import Path

from cfun.font import get_chinese_font_path_random, get_chinese_font_paths


def test_font():
    import pprint

    fonts_info = get_chinese_font_paths()
    print(f"系统平台: {fonts_info['platform']}")
    print("中文字体路径:")
    pprint.pprint(fonts_info["chinese"])
    print("随机中文字体路径:")
    fontpath = get_chinese_font_path_random()
    assert isinstance(fontpath, Path), f"Expected Path, got {type(fontpath)}"
    print(type(fontpath), fontpath)


if __name__ == "__main__":
    test_font()
