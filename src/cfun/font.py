import glob
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


def get_chinese_font_path_random(key="song") -> Optional[Path]:
    """
    随机获取一个中文字体文件路径, 优先songti

    Args:
        key (str): 字体类型关键字,默认是 "song",可以是 "song" 或 "hei" 等

    Returns:
        Optional[Path]: 随机中文字体文件路径,如果没有找到则返回 None

    Example:
        ```python
        from cfun.font import get_chinese_font_path_random
        font_path = get_chinese_font_path_random()
        print(font_path)
        ```
    """
    try:
        # 尝试使用系统字体
        font_paths = get_chinese_font_paths()["chinese"]
        # 优先选择宋体
        assert isinstance(font_paths, list), "font_paths should be a list"
        songti_fonts = [Path(p) for p in font_paths if key in p.name.lower()]
        if songti_fonts:
            # 路径比较短的
            songti_fonts = sorted(songti_fonts, key=lambda p: len(str(p)))
            # 选择第一个
            return songti_fonts[0]
        elif font_paths:
            return random.choice(font_paths)
    except Exception:
        pass
    return None


def get_chinese_font_paths() -> Dict[str, object]:
    """
    获取当前系统中的中文字体文件路径（支持 Windows、Linux、macOS,自动判断是否可用 fc-list 命令来查找字体

    Returns:
        Dict[str, object]: 包含平台、系统类型与字体路径
    """
    system = platform.system().lower()
    font_paths = []

    chinese_keywords = [
        "simsun",
        "simhei",
        "simfang",
        "simkai",
        "sourcehansans",
        "song",
        "hei",
        "kai",
        "fangsong",
        "ming",
        "cjk",
        "microsoft yahei",
        "pingfang",
        "noto sans cjk",
        "source han",
        "华文",
        "方正",
        "思源",
        "文泉驿",
    ]
    chinese_keywords = [kw.lower() for kw in chinese_keywords]

    if has_fc_list():
        font_paths = _fc_list_paths()
    elif system == "windows":
        font_paths = _windows_font_paths(chinese_keywords)
    elif system == "darwin":
        font_paths = _macos_font_paths(chinese_keywords)
    elif system == "linux":
        font_paths = _linux_font_paths(chinese_keywords)

    # 去重 + 转换为 Path 对象

    cleaned = [Path(p) for p in font_paths if Path(p).exists() and Path(p).is_file()]

    return {
        "platform": system,
        "system": system,
        "chinese": cleaned,
    }


def get_fixed_fonts(chinese="song", english="arial") -> Dict[str, Optional[Path]]:
    """
    获取固定的中英文字体路径，
    根据系统平台自动选择合适的字体路径。
    Args:
        chinese (str): 中文字体关键字, 默认是 "song" (包含 song 的字体, 不区分大小写)
        english (str): 英文字体关键字, 默认是 "arial" (包含 arial 的字体, 不区分大小写)

    Returns:
        Dict[str, Optional[Path]]: 包含中文和英文字体路径的字典
            {
                "chinese": Path对象或None,
                "english": Path对象或None
            }

    Example:
        ```python
        from cfun.font import get_fixed_fonts
        fonts = get_fixed_fonts()
        print(fonts["chinese"])  # 输出中文字体路径
        print(fonts["english"])  # 输出英文字体路径
        ```
    """
    system = platform.system().lower()

    chinese_keywords = [chinese]
    english_keywords = [english]

    # 根据平台获取字体路径列表
    if system == "windows":
        chinese_font = _windows_font_paths(chinese_keywords)
        english_font = _windows_font_paths(english_keywords)
    elif system == "darwin":
        chinese_font = _macos_font_paths(chinese_keywords)
        english_font = _macos_font_paths(english_keywords)
    elif system == "linux":
        chinese_font = _linux_font_paths(chinese_keywords)
        english_font = _linux_font_paths(english_keywords)
    else:
        return {
            "chinese": None,
            "english": None,
        }
    english_font.sort(key=lambda p: Path(p).stem.split()[-1])  # 按照字体名称排序
    chinese_font.sort(key=lambda p: Path(p).stem.split()[-1])  # 按照字体名称排序
    return {
        "chinese": Path(chinese_font[0]) if chinese_font else None,
        "english": Path(english_font[0]) if english_font else None,
    }


def has_fc_list() -> bool:
    """判断是否可以使用 fc-list 命令"""
    try:
        subprocess.check_output(["fc-list", "--version"], stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _fc_list_paths() -> List[str]:
    """使用 fc-list 获取支持中文的字体路径"""
    try:
        output = subprocess.check_output(["fc-list", ":lang=zh", "file"], text=True)
        paths = []
        for line in output.splitlines():
            path = line.strip().rstrip(":")  # 去除尾部冒号
            if path and Path(path).is_file():
                paths.append(path)
        return sorted(set(paths))
    except Exception as e:
        print(f"[fc-list error] {e}")
        return []


def _windows_font_paths(keywords: List[str]) -> List[str]:
    """从 Windows 注册表中获取字体路径"""
    paths = []
    try:
        import winreg

        fonts_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts",
        )
        i = 0
        while True:
            try:
                name, value, _ = winreg.EnumValue(key, i)
                i += 1
                if any(kw in name.lower() for kw in keywords):
                    full_path = os.path.join(fonts_dir, value)
                    if os.path.exists(full_path):
                        paths.append(full_path)
            except OSError:
                break
    except Exception:
        pass
    return paths


def _macos_font_paths(keywords: List[str]) -> List[str]:
    """macOS 下从常见字体目录中查找含有中文关键词的字体"""
    paths = []
    font_dirs = [
        "/System/Library/Fonts/Supplemental",
        "/System/Library/Fonts",
        "/Library/Fonts",
        os.path.expanduser("~/Library/Fonts"),
    ]
    try:
        for font_dir in font_dirs:
            for ext in ("*.ttf", "*.ttc", "*.otf", "*.TTF", "*.OTF", "*.TTC"):
                for font_path in glob.glob(os.path.join(font_dir, ext)):
                    if any(kw in font_path.lower() for kw in keywords):
                        paths.append(font_path)
    except Exception:
        pass
    return paths


def _linux_font_paths(keywords: List[str]) -> List[str]:
    """Linux 上从常见字体路径中查找字体（备用于没有 fc-list 的情况）"""
    paths = []
    font_dirs = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
    ]
    try:
        for font_dir in font_dirs:
            for ext in ("*.ttf", "*.ttc", "*.otf", "*.TTF", "*.OTF", "*.TTC"):
                for font_path in glob.glob(
                    os.path.join(font_dir, "**", ext), recursive=True
                ):
                    if any(kw in font_path.lower() for kw in keywords):
                        paths.append(font_path)
    except Exception:
        pass
    return paths


if __name__ == "__main__":
    import pprint

    fonts_info = get_chinese_font_paths()
    print(f"系统平台: {fonts_info['platform']}")
    print("中文字体路径:")
    pprint.pprint(fonts_info["chinese"])
    print("随机中文字体路径:")
    fontpath = get_chinese_font_path_random()
    print(type(fontpath), fontpath)

    fixed_fonts = get_fixed_fonts()
    print("固定中文字体:", fixed_fonts["chinese"])
    print("固定英文字体:", fixed_fonts["english"])
