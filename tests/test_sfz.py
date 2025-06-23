from pathlib import Path

import pandas as pd

from cfun.sfz.idcardocr import IDCardOCR


# 示例用法
def test_sfz():
    from cfundata import cdata

    model_path = cdata.SFZ_DET_ONNX
    idcard = IDCardOCR(model_path)  # 创建sfz处理器实例

    images = [f for f in Path("idcard").rglob("*.jpg") if f.is_file()]
    print(f"Found {len(images)} images.")
    images = sorted(images)
    df = []
    for img_path in images:
        try:
            info = idcard.process_image(img_path)
            df.append(info)
            # print(f"Processed {img_path}:\n{info}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            break
    df = pd.DataFrame(df)
    df.fillna("", inplace=True)
    print(df)
    print("All images processed.")


if __name__ == "__main__":
    test_sfz()
