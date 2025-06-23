import hashlib
import os
import shutil
from pathlib import Path
from typing import Optional

import ddddocr
from loguru import logger
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


def _is_image_valid(filepath: Path) -> bool:
    try:
        with Image.open(filepath) as img:
            img.verify()  # 验证图片文件本身
        with Image.open(filepath) as img:
            img.load()  # 尝试加载图像数据
        return True
    except Exception as e:
        logger.info(f"Invalid image: {filepath}, error: {e}")
        return False


def _calculate_md5(file_path: Path) -> str:
    hash_md5 = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _remove_alpha(img: Image.Image, background_color=(255, 255, 255)) -> Image.Image:
    """去除 alpha 通道并转为 RGB"""
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, background_color)
        img = img.convert("RGBA")
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def _resize_box_to(
    box, img_width, img_height, size=60, min_size=10
) -> Optional[list[int]]:
    """
    将 box 调整为指定大小（正方形），确保不超出图像边界。

    Args:
        box: 原始框 [x1, y1, x2, y2]
        img_width: 图像宽度
        img_height: 图像高度
        size: 目标裁剪框大小（正方形边长）
        min_size: 最小允许原始框的宽或高，小于此值将被忽略

    Returns:
        调整后的框坐标 [x1, y1, x2, y2]，如过小则返回 None
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1

    if w < min_size or h < min_size:
        return None  # 跳过太小的框

    # 中心点
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    half = size / 2

    # 裁剪框，限制在图像边界内
    new_x1 = max(0, int(cx - half))
    new_y1 = max(0, int(cy - half))
    new_x2 = min(img_width, int(cx + half))
    new_y2 = min(img_height, int(cy + half))

    # 如果因为边界限制导致大小太小，也跳过
    if new_x2 - new_x1 < min_size or new_y2 - new_y1 < min_size:
        return None

    return [new_x1, new_y1, new_x2, new_y2]


def main1_unique_images(
    image_dir: Path,
    output_dir: Path,
    source_icons_dir: Optional[Path] = None,
    output_icons_dir: Optional[Path] = None,
    image_suffix: Optional[list[str]] = None,
):
    """
    根据 MD5 去重图像,并(可选)同步复制配套icon文件

    Args:
        image_dir (Path): 输入图像目录,含子文件
        output_dir (Path): 去重后的图像输出目录
        source_icons_dir (Optional[Path]): 配套icon文件的源目录,可为 None, 如果存在, 则 images 文件的 stem 必须包含在 icon文件名中
        output_icons_dir (Optional[Path]): 配套icon的目标目录,可为 None
        image_suffix (Optional[list[str]]): 图像文件后缀列表,默认为 [".png", ".jpg", ".jpeg", ".webp"]
    """
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if source_icons_dir and output_icons_dir:
        if not source_icons_dir.exists():
            raise FileNotFoundError(
                f"Source icons directory not found: {source_icons_dir}"
            )
        if Path(output_icons_dir).exists():
            shutil.rmtree(output_icons_dir)
        output_icons_dir.mkdir(parents=True, exist_ok=True)
    imgfiles = [f for f in image_dir.rglob("*") if f.is_file()]
    if image_suffix is None:
        image_suffix = [".png", ".jpg", ".jpeg", ".webp"]
    image_suffix = [s.lower() for s in image_suffix]  # 统一小写

    imgfiles = [f for f in imgfiles if f.suffix.lower() in image_suffix]

    logger.info(f"查找图像文件后缀: {image_suffix}")
    logger.info(f"在 {image_dir} 中找到 {len(imgfiles)} 张图像文件")
    # 过滤掉无效图像
    imgfiles = [f for f in imgfiles if _is_image_valid(f)]

    md5_map = {}
    for imgfile in imgfiles:
        md5 = _calculate_md5(imgfile)
        if md5 not in md5_map:
            md5_map[md5] = imgfile

    logger.info(f"✅ 找到 {len(md5_map)} 张不重复图像")

    for _, imgfile in md5_map.items():
        relative_path = Path(imgfile).relative_to(image_dir)
        dest_img = output_dir / relative_path
        dest_img.parent.mkdir(parents=True, exist_ok=True)

        if not dest_img.exists():
            shutil.copy(imgfile, dest_img)

        # 如果提供了配套icon目录, 则复制相关的icon文件
        if source_icons_dir and output_icons_dir:
            stem = imgfile.stem
            matches = [f for f in source_icons_dir.rglob("*") if f.is_file()]
            matches = [
                f
                for f in matches
                if stem in f.stem and f.suffix.lower() in image_suffix
            ]

            for match in matches:
                # 复制到目标目录
                dest_icons = output_icons_dir / match.relative_to(source_icons_dir)
                dest_icons.parent.mkdir(parents=True, exist_ok=True)
                if not dest_icons.exists():
                    shutil.copy(match, dest_icons)

    logger.info(" 去重和复制完成")
    # 统计新的图像数量
    unique_images = list(output_dir.rglob("*"))
    logger.info(f"✅ 去重后图像数量: {len(unique_images)}")
    # 统计icon数量
    if output_icons_dir:
        unique_icons = list(output_icons_dir.rglob("*"))
        logger.info(f"✅ 去重后配套icon数量: {len(unique_icons)}")


def main2_crop_images(
    bg_dir: Path,
    icon_dir: Path,
    output_dir: Path,
    box_size: int = 60,
    clear_output: bool = True,
    image_suffix: Optional[list[str]] = None,
    iscopybg: bool = False,
):
    """
    剪切底图中的 DET 区域，并整理对应的前景图

    Args:
        bg_dir: 底图所在目录，唯一命名
        icon_dir: 前景图目录，命名中包含底图 stem
        output_dir: 输出目录
        box_size: 裁剪框大小（正方形）
        clear_output: 是否清空旧的输出目录
        image_suffix: 图像文件后缀列表，默认为 [".png", ".jpg", ".jpeg", ".webp"]
        iscopybg: 是否复制底图到输出目录, 默认为 False, 如果为 True, 只是方便查看
    """
    logger.info("开始处理图片...")

    det = ddddocr.DdddOcr(det=True, show_ad=False)

    if clear_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if image_suffix is None:
        image_suffix = [".png", ".jpg", ".jpeg", ".webp"]
    image_suffix = [s.lower() for s in image_suffix]  # 统一小写
    bg_files = [
        f for f in bg_dir.rglob("*") if f.is_file() and f.suffix.lower() in image_suffix
    ]
    icon_files = [
        f
        for f in icon_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in image_suffix
    ]

    logger.info(f"🖼️ 发现 {len(bg_files)} 张底图, {len(icon_files)} 张前景图")

    for idx, bg_file in enumerate(bg_files):
        if idx % 100 == 0:
            logger.info(f"🔄 进度 {idx}/{len(bg_files)}")

        try:
            with bg_file.open("rb") as f:
                image_bytes = f.read()
            bboxes = det.detection(image_bytes)
        except Exception as e:
            logger.warning(f"⚠️ 跳过损坏图片 {bg_file}: {e}")
            continue

        subdir = _calculate_md5(bg_file)[:10]
        related_icons = [f for f in icon_files if bg_file.stem in f.stem]

        if not related_icons:
            logger.warning(f"⚠️ 找不到前景图: {bg_file.name}")
            continue

        if _is_image_valid(bg_file):
            try:
                img = Image.open(bg_file)
            except UnidentifiedImageError as e:
                logger.warning(f"⚠️ 无法打开图片 {bg_file}: {e}")
                continue
        else:
            logger.warning(f"⚠️ 无效图片 {bg_file}, 跳过")
            continue

        # 处理所有检测框
        width, height = img.size
        resized_boxes = [
            _resize_box_to(box, width, height, size=box_size) for box in bboxes
        ]
        resized_boxes = [box for box in resized_boxes if box is not None]

        for i, (x1, y1, x2, y2) in enumerate(resized_boxes):
            cropped = img.crop((x1, y1, x2, y2))
            crop_path = output_dir / subdir / str(i) / f"{bg_file.stem}_cropped_{i}.png"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop_path = output_dir / subdir / f"{bg_file.stem}_cropped_{i}.png"
            cropped.save(crop_path)
        if iscopybg:
            # 复制底图到输出目录
            dest_bg_path = output_dir / subdir / bg_file.name
            dest_bg_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(bg_file, dest_bg_path)
        # 拷贝相关前景图
        for icon_file in related_icons:
            dest_path = output_dir / subdir / icon_file.name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(icon_file, dest_path)

    logger.success("✅ 所有图片处理完成！")


def main3_class_images(
    source_dir: Path,
    target_dir: Path,
    isrotate: bool = True,
    rotate_range: tuple[int, int] = (-30, 30),
    rotate_step: int = 3,
    background_color: Optional[tuple[int, int, int]] = None,
    image_suffix: Optional[list[str]] = None,
    subdir_min_imglen: int = 2,
    subdir_max_imglen: int = 3,
    unique_kw: Optional[str] = "rgba",
    unique_start: Optional[str] = None,
    unique_end: Optional[str] = None,
) -> None:
    """
    处理分类图像：去 alpha、旋转、保存到目标目录、去重。

    Args:
        source_dir: 原始分类图片目录
        target_dir: 输出目录
        isrotate: 是否生成旋转图像，默认 True
        rotate_range: 旋转角度范围 (start, end)，默认 (-30, 30)
        rotate_step: 旋转角度步长，默认 3
        background_color: 旋转用于填充 alpha 的背景色, 默认为白色 (255, 255, 255)
        image_suffix: 图像文件后缀列表，默认为 [".png", ".jpg", ".jpeg", ".webp"]
        subdir_min_imglen: 子目录下最小图像数量，默认 2
        subdir_max_imglen: 子目录下最大图像数量，默认 3, 不在这个范围内的子目录将被忽略
        unique_kw: 用于唯一标识的关键字，默认为 "rgba", 透明图像将被作为唯一标识
        unique_start: 可选的唯一标识开始字符串
        unique_end: 可选的唯一标识结束字符串
    """
    if Path(target_dir).exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if background_color is None:
        background_color = (255, 255, 255)  # 默认白色背景

    if image_suffix is None:
        image_suffix = [".png", ".jpg", ".jpeg", ".webp"]
    image_suffix = [s.lower() for s in image_suffix]  # 统一小写

    subdirs = [d for d in source_dir.rglob("*") if d.is_dir()]
    md5map = []
    # 对每个子目录进行处理
    for sub in subdirs:
        imgfiles = [
            f
            for f in sub.rglob("*")
            if f.is_file() and f.suffix.lower() in image_suffix
        ]
        if not imgfiles:
            continue
        if len(imgfiles) < subdir_min_imglen or len(imgfiles) > subdir_max_imglen:
            continue

        target_dir.mkdir(parents=True, exist_ok=True)

        # 寻找每个目录下的唯一标识, 这里先用透明图像作为唯一标识
        unique_id = []
        for imgfile in imgfiles:
            img = Image.open(imgfile)
            if unique_kw == "rgba":
                # 检测 RGBA 图像
                if img.mode == "RGBA":
                    unique_id.append(imgfile)
            elif unique_start and not unique_end:
                # 以某些文件开头
                if imgfile.name.startswith(unique_start):
                    unique_id.append(imgfile)
            elif unique_end and not unique_start:
                # 以某些文件结尾
                if imgfile.name.endswith(unique_end):
                    unique_id.append(imgfile)
            elif unique_start and unique_end:
                # 以某些文件开头和结尾
                if imgfile.name.startswith(unique_start) and imgfile.name.endswith(
                    unique_end
                ):
                    unique_id.append(imgfile)

        # if len(unique_id) == 0:
        #     logger.info(f"❗ 子目录 {sub} 中没有符合条件的唯一标识图像，跳过")
        #     continue
        assert len(unique_id) > 0, (
            f"❗子目录 {sub} 中没有符合条件的唯一标识图像, 请检查"
        )

        # 检测到是唯一标识图像,# 把md5 到做目录
        md5 = _calculate_md5(unique_id[0])
        md5map.append(md5)

        # 处理每个子目录下的图片
        newsubdir = target_dir / md5[:10]
        newsubdir.mkdir(parents=True, exist_ok=True)
        for imgfile in imgfiles:
            img = Image.open(imgfile)
            newpath = newsubdir / imgfile.name
            if img.mode == "RGBA":
                img = _remove_alpha(img, background_color)
                img.save(newpath, format="PNG")
                # 生成旋转图像
                if isrotate:
                    for idx, angle in enumerate(
                        range(rotate_range[0], rotate_range[1] + 1, rotate_step)
                    ):
                        rotated = img.rotate(
                            angle,
                            expand=False,
                            resample=Image.Resampling.BICUBIC,
                            fillcolor=background_color,
                        )
                        rotated_path = newpath.with_name(
                            f"{imgfile.stem}_angle{idx}.png"
                        )
                        rotated.save(rotated_path, format="PNG")
            else:
                # 直接保存到目标目录
                shutil.copy2(imgfile, newpath)

    md5map = list(set(md5map))  # 去重 MD5 列表
    logger.info(f"✅ 处理完成，共处理 {len(md5map)} 个分类(唯一标识)")
    logger.info("✅ 图像处理完成, 开始对每个分类进行去重")
    subdirs = [d for d in target_dir.rglob("*") if d.is_dir()]
    if not subdirs:
        return
    for sub in subdirs:
        subimgfiles = [
            f
            for f in sub.rglob("*")
            if f.is_file() and f.suffix.lower() in image_suffix
        ]
        subimgfiles.sort()
        submd5 = []
        for f in subimgfiles:
            if not f.is_file():
                continue
            try:
                md5 = _calculate_md5(f)
                if md5 in submd5:
                    os.remove(f)
                else:
                    submd5.append(md5)
            except Exception as e:
                logger.info(f"❌ 无法计算 MD5：{f}，错误：{e}")

    logger.info("🎉 所有图片处理与去重完成")

    # 统计新的图像数量
    unique_images = list(target_dir.rglob("*"))
    unique_images = [f for f in unique_images if f.is_file()]
    logger.info(f"✅ 去重后图像数量: {len(unique_images)}")


def main4_check(
    model_path: str,
    img_dir: str | Path,
    verbose: bool = False,
    img_suffix: Optional[list[str]] = None,
) -> None:
    """
    检查图片是否被正确分类。

    Args:
        model_path (str): YOLOv8 模型路径，如 'best.pt', 必须是 YOLO 模型文件
        img_dir (str | Path): 图像目录，每个子目录为类别名
        verbose (bool): 是否打印分类错误的信息
        img_suffix (Optional[list[str]]): 图像文件后缀列表，默认为 [".png", ".jpg", ".jpeg", ".webp"]
    """
    model = YOLO(model_path)
    img_dir = Path(img_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    if img_suffix is None:
        img_suffix = [".png", ".jpg", ".jpeg", ".webp"]
    img_suffix = [s.lower() for s in img_suffix]  # 统一小

    all_images = [
        f for f in img_dir.rglob("*") if f.is_file() and f.suffix.lower() in img_suffix
    ]

    logger.info(f"Found {len(all_images)} images in {img_dir}")

    for i in all_images:
        try:
            img = Image.open(i)
            img = _remove_alpha(img)
            result = model(img, verbose=False)
            all_names = result[0].names  ##  类别字典
            top1 = result[0].probs.top1  # 最大概率对应的索引
            top1name = all_names[top1]  # 最大概率对应的类别
            ecls = i.parent.stem
            if top1name != ecls:
                if verbose:
                    logger.info(f"Image {i} classified as {top1name}, expected {ecls}")
        except Exception as e:
            logger.info(f"⚠️ Error processing {i}: {e}")


if __name__ == "__main__":
    pass
    # main1_unique_images(
    #     image_dir=Path("icon4_imgs"),
    #     output_dir=Path("imgs_test"),
    #     source_icons_dir=Path("icon4_ques"),
    #     output_icons_dir=Path("imgs_test_ques"),
    # )

    # main2_crop_images(
    #     bg_dir=Path("imgs_test"),
    #     icon_dir=Path("imgs_test_ques"),
    #     output_dir=Path("imgs_test_cropped"),
    #     box_size=60,
    #     clear_output=True,
    #     image_suffix=[".png", ".jpg", ".jpeg", ".webp"],
    # )
    # main3_class_images(
    #     source_dir=Path("unique"), target_dir=Path("imgs_test_classified__remove_alpha")
    # )
    # main4_check(
