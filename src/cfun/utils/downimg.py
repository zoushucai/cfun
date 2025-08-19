from pathlib import Path
from typing import Sequence, Union

import requests


def download_img(
    url: Union[str, Sequence[str]], path: Union[str, Sequence[str]], timeout: int = 15
) -> dict[str, list[str]]:
    """
    下载图片 (使用 requests，可自定义 header)
    """
    urls = [url] if isinstance(url, str) else list(url)
    paths = [path] if isinstance(path, str) else list(path)

    if len(urls) != len(paths):
        raise ValueError("url 和 path 长度不一致")

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
    }
    result = {"success": [], "failed": []}

    for u, p in zip(urls, paths, strict=True):
        try:
            resp = requests.get(u, headers=headers, timeout=timeout, stream=True)
            resp.raise_for_status()
            save_path = Path(p).expanduser().resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            result["success"].append(str(save_path))
        except Exception as e:
            print(f"下载失败: {u} -> {p}, 错误: {e}")
            result["failed"].append((u, str(p)))
    return result
