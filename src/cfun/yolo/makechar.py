import random
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..font import get_chinese_font_path_random


class MakeCharImage:
    """生成单字符图片

    Attributes:
        text (str): 生成的字符
        image_size (Tuple[int, int]): 图片大小
        offset (Union[int, float]): 偏移量,用于加粗文字
        font_path (str): 字体文件路径, None表示随机获取
        output_path (Optional[Union[str, Path]]): 输出图片路径
        noise_density (float): 噪声密度,范围 [0, 1]

    Example:
        ```python
        from cfun.yolo.makechar import MakeCharImage
        generator = MakeCharImage(
            text="好",
            image_size=(64, 64),
            offset=0.5,
            font_path=None,
            output_path="output/A.png",
            noise_density=0.25,
        )
        generator.generate_image()
        generator.save_image()
        ```
    """

    def __init__(
        self,
        text: str,
        image_size: Tuple[int, int] = (64, 64),
        offset: Union[int, float] = 0,
        font_path: str | Path | None = None,
        output_path: str | Path | None = None,
        noise_density: float = 0.25,
        bg_color: Optional[Tuple[int, int, int]] = None,
        fg_color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        assert len(text) == 1, "text must be a single character"
        self.text = text
        self.image_size = image_size
        self.offset = offset
        self.font_path = font_path if font_path else str(get_chinese_font_path_random())
        if not self.font_path:
            raise ValueError("Font path is required.")
        self.output_path = Path(output_path) if output_path else None
        self.noise_density = noise_density
        self.generated_image = None
        self.bg_color = bg_color if bg_color else self.random_bg_color()
        self.fg_color = fg_color if fg_color else self.random_text_color()

    @staticmethod
    def load_font(font_path: str, font_size: int) -> Optional[ImageFont.FreeTypeFont]:
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"[Font Load Error] Could not load font {font_path}: {e}")
            return None

    @staticmethod
    def random_bg_color() -> Tuple[int, int, int]:
        return (
            random.randint(200, 255),
            random.randint(200, 255),
            random.randint(200, 255),
        )  # 偏白背景

    @staticmethod
    def random_text_color() -> Tuple[int, int, int]:
        return (
            random.randint(0, 150),
            random.randint(0, 150),
            random.randint(0, 150),
        )  # 偏黑文字

    def draw_bold_text(self, draw, x, y, font, color):
        for dx in [-self.offset, 0, self.offset]:
            for dy in [-self.offset, 0, self.offset]:
                draw.text(
                    (x + dx, y + dy), self.text, font=font, fill=color, anchor="mm"
                )

    def add_noise(self, image: Image.Image) -> Image.Image:
        arr = np.array(image)
        noise_mask = np.random.rand(*arr.shape[:2]) < self.noise_density
        noise = np.random.randint(0, 256, size=arr.shape, dtype=np.uint8)
        arr[noise_mask] = noise[noise_mask]
        return Image.fromarray(arr)

    def generate_image(self) -> Image.Image:
        """做图"""
        font_size = int(min(self.image_size) * 0.85)  # 字体大小
        font = self.load_font(str(self.font_path), font_size)
        if not font:
            raise ValueError(f"Could not load font from: {self.font_path}")

        bg_color = self.bg_color
        text_color = self.fg_color

        # 创建背景（直接使用目标尺寸,避免后续计算问题）
        image = Image.new("RGB", self.image_size, bg_color)
        draw = ImageDraw.Draw(image)

        # 计算文本绘制中心点
        center_x = image.width // 2
        center_y = image.height // 2

        # 加粗文字（如果设置了 offset）
        if self.offset > 0:
            self.draw_bold_text(draw, center_x, center_y, font, text_color)
        else:
            draw.text(
                (center_x, center_y), self.text, font=font, fill=text_color, anchor="mm"
            )

        image = self.add_noise(image)
        self.generated_image = image
        return image

    def save_image(self, path: Optional[Union[str, Path]] = None):
        """保存图片

        Args:
            path (Optional[Union[str, Path]]): 保存路径,默认使用初始化时的 output_path
        """
        if self.generated_image is None:
            self.generate_image()
        output = Path(path) if path else self.output_path
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            if self.generated_image is not None:
                self.generated_image.save(str(output))


if __name__ == "__main__":
    fonts_path = [f for f in Path("fonts").glob("*")]
    print("Fonts found:", fonts_path)
    generator = MakeCharImage(
        text="好",
        image_size=(64, 64),
        offset=0.5,
        font_path=str(fonts_path[0]),
        output_path="output/A1.png",
        noise_density=0.25,
    )
    generator.generate_image()
    generator.save_image()

    # 单字符测试
    # img_dir = Path("train")
    # subdirs = [d for d in img_dir.iterdir() if d.is_dir()]
    # # 对subdirs 按照拼音排序
    # # 这里可以使用拼音排序的库,比如 pypinyin
    # import pypinyin
    # subdirs.sort(key=lambda x: pypinyin.lazy_pinyin(x.name))
    # for idx, subdir in enumerate(subdirs):
    #     if idx % 10 == 0:
    #         print(f"Processing {idx}/{len(subdirs)}: {subdir.name}")
    #     name = subdir.name
    #     for font in fonts_path:
    #         fname = f"{name}_{font.stem}.png".replace(" ", "_").replace("-", "_")
    #         fname = fname.lower()

    #         generator = MakeCharImage(
    #             text=name,
    #             image_size=(64, 64),
    #             offset=0.4,
    #             font_path=str(font),
    #             output_path=f"{subdir}/{fname}",
    #             noise_density=0.25,
    #         )
    #         generator.generate_image()
    #         generator.save_image()
