import hashlib
from hashlib import md5
from pathlib import Path

import requests

from .dddocrtool import ImageDet

imgdet = ImageDet()


class Chaojiying_Client:
    """
    超级鹰验证码客服端

    参考文档：http://www.chaojiying.com/api.php

    Attributes:
        username (str): 超级鹰用户名
        password (str): 超级鹰密码
        soft_id (str): 超级鹰软件ID
    """

    def __init__(self, username, password, soft_id):
        """初始化超级鹰客户端

        Args:
            username (str): 超级鹰用户名
            password (str): 超级鹰密码
            soft_id (str): 超级鹰软件ID
        """

        self.username = username
        password = password.encode("utf8")
        self.password = md5(password).hexdigest()
        self.soft_id = soft_id
        self.base_params = {
            "user": self.username,
            "pass2": self.password,
            "softid": self.soft_id,
        }
        self.headers = {
            "Connection": "Keep-Alive",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0",
        }

    @staticmethod
    def calculate_md5(file_path: Path | str) -> str:
        """计算文件的 MD5 值

        Args:
            file_path (str): 文件路径

        Returns:
            str: 文件的 MD5 值
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        assert file_path.exists(), f"文件不存在：{file_path}"
        assert file_path.is_file(), f"路径不是文件：{file_path}"
        assert file_path.stat().st_size > 0, f"文件大小为0：{file_path}"
        hash_md5 = hashlib.md5()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @staticmethod
    def read_image(image_path: Path | str) -> bytes:
        """读取图片文件并返回字节

        Args:
            image_path (str): 图片路径

        Returns:
            bytes: 图片字节
        """
        if isinstance(image_path, str):
            image_path = Path(image_path)
        assert image_path.is_file(), f"路径不是文件：{image_path}"
        assert image_path.stat().st_size > 0, f"文件大小为0：{image_path}"
        assert Path(image_path).exists(), f"文件不存在：{image_path}"

        with open(image_path, "rb") as f:
            image = f.read()
        return image

    def PostPic(self, im, codetype) -> dict:
        """通过网络上传图片进行识别

        Args:
            im (bytes): 图片字节
            codetype (int): 题目类型 参考 http://www.chaojiying.com/price.html

        Returns:
            dict: 超级鹰返回的结果
        """
        params = {
            "codetype": codetype,
        }
        params.update(self.base_params)
        files = {"userfile": ("ccc.jpg", im)}
        r = requests.post(
            "http://upload.chaojiying.net/Upload/Processing.php",
            data=params,
            files=files,
            headers=self.headers,
        )
        return r.json()

    def PostPic_base64(self, base64_str, codetype) -> dict:
        """通过网络上传图片进行识别

        Args:
            base64_str (str): 图片的base64编码字符串
            codetype (int): 题目类型 参考 http://www.chaojiying.com/price.html

        Returns:
            dict: 超级鹰返回的结果
        """
        params = {"codetype": codetype, "file_base64": base64_str}
        params.update(self.base_params)
        r = requests.post(
            "http://upload.chaojiying.net/Upload/Processing.php",
            data=params,
            headers=self.headers,
        )
        return r.json()

    def ReportError(self, im_id) -> dict:
        """上传错误的图片ID进行反馈

        Args:
            im_id (str): 图片ID

        Returns:
            dict: 超级鹰返回的结果
        """
        params = {
            "id": im_id,
        }
        params.update(self.base_params)
        r = requests.post(
            "http://upload.chaojiying.net/Upload/ReportError.php",
            data=params,
            headers=self.headers,
        )
        return r.json()

    def parse_response(self, res: dict | str) -> list:
        """解析超级鹰返回的字符串格式

        Args:
            res (str | dict): 超级鹰返回的字符串 或者字典格式

        Returns:
            list: 解析后的列表
                [
                    {"name": "之", "coordinates": [207, 115]},
                    {"name": "成", "coordinates": [158, 86]},
                    {"name": "人", "coordinates": [126, 44]}
                ]

        Example:
            ```python
            input_str = '之,207,115|成,158,86|人,126,44' #或者直接输入字典格， 会提取里面的 pic_str字段
            output = parse_string_chaojiying(input_str)
            print(output)
            ```
        """

        if isinstance(res, dict):
            res = res["pic_str"]

        assert isinstance(res, str), "超级鹰识别失败,请检查图片是否正确"
        assert "|" in res, "超级鹰识别失败,请检查图片是否正确"
        items = res.split("|")

        result = []
        for item in items:
            # 将每个部分再以 "," 分隔
            parts = item.split(",")
            name = parts[0]  # 第一个部分是 name
            x = int(parts[1])  # 第二个部分是 x 坐标
            y = int(parts[2])

            # 创建字典对象并添加到列表
            result.append({"name": name, "coordinates": [x, y]})

        return result

    def get(self, image_path: str, codetype: int = 9800) -> list:
        """根据图片和题目类型获取超级鹰识别结果，

        根据图片和题目类型获取超级鹰识别结果， 暂时只针对 codetype=9800 进行处理，其他类型的暂未测试

        Args:
            image_path (str): 图片路径
            codetype (int): 题目类型 参考 http://www.chaojiying.com/price.html

        Returns:
            list: 扩展后的结果


        Example:
            ```python
            from cfun.yzm.chaojiying import Chaojiying_Client
            from cfun.yzm.dddocrtool import ImageDet
            chaojiying = Chaojiying_Client("****", "****", "***")
            image_path = Path(__file__).parent / "images" / "image_detect_01.png"
            result = chaojiying.get(image_path, 9800)
            print(result) #封装后的结果
            '''
            [
                {
                    "name": "之",
                    "coordinates": [207, 115],
                    "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "image_width": 100,
                    "image_height": 100
                },
                {
                    "name": "成",
                    "coordinates": [158, 86],
                    "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "image_width": 100,
                    "image_height": 100
                },
                {
                    "name": "人",
                    "coordinates": [126, 44],
                    "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "image_width": 100,
                    "image_height": 100
                }
            ]
            '''
            ```
        """
        im = self.read_image(image_path)
        res = self.PostPic(im, codetype)

        result = self.parse_response(res)
        extended_data = imgdet.extendCJYRecognition(result, image_path)

        # 计算图片的 md5 值
        md5_value = self.calculate_md5(image_path)
        # 将 md5 值添加到结果中
        for item in extended_data:
            item["rawimgmd5"] = md5_value
        return extended_data


if __name__ == "__main__":
    pass
    # chaojiying = Chaojiying_Client("*****", "*****", "*****")
    # image_path = Path(__file__).parent / "images" / "image_detect_01.png"
    # result = chaojiying.get(image_path, 9800)
    # print(result)
