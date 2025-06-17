"""
识别sfz正面, 传入图片路径, 返回识别结果, (代码来源网上, 效果不错)
"""

import re
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np

from ..yolo.detect import Detector


class IdCardStraight:
    """
    sfzOCR返回结果,正常的sfz大概率识别没有什么问题,少数名族sfz,壮文、藏文、蒙文基本识别也没问题
    """

    nation_list = [
        "汉",
        "蒙古",
        "回",
        "藏",
        "维吾尔",
        "苗",
        "彝",
        "壮",
        "布依",
        "朝鲜",
        "满",
        "侗",
        "瑶",
        "白",
        "土家",
        "哈尼",
        "哈萨克",
        "傣",
        "黎",
        "傈僳",
        "佤",
        "畲",
        "高山",
        "拉祜",
        "水",
        "东乡",
        "纳西",
        "景颇",
        "柯尔克孜",
        "土",
        "达斡尔",
        "仫佬",
        "羌",
        "布朗",
        "撒拉",
        # "毛难",
        "毛南",  # 同"毛难",
        "仡佬",
        "锡伯",
        "阿昌",
        "普米",
        "塔吉克",
        "怒",
        "乌孜别克",
        "俄罗斯",
        "鄂温克",
        "德昂",
        "保安",
        "裕固",
        "京",
        "塔塔尔",
        "独龙",
        "鄂伦春",
        "赫哲",
        "门巴",
        "珞巴",
        "基诺",
    ]

    def __init__(self, result):
        self.result = result
        self.result0 = deepcopy(result)
        self.out = {"result": {}}
        self.res = self.out["result"]
        self.res["name"] = ""
        self.res["idNumber"] = ""
        self.res["address"] = ""
        self.res["gender"] = ""
        self.res["nationality"] = ""

    @staticmethod
    def verifyByIDCard(idcard):
        """
        验证sfz号码是否有效
        """
        if len(idcard) != 18:
            return False

        weight = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
        validate = ["1", "0", "X", "9", "8", "7", "6", "5", "4", "3", "2"]
        total_sum = sum(weight[i] * int(idcard[i]) for i in range(len(weight)))
        return validate[total_sum % 11] == idcard[-1]

    def birth_no(self):
        """
        提取并验证sfz号码,识别性别。
        """
        for txt in self.result:
            # 使用统一的正则表达式匹配18位sfz（末位可能是 X 或 x）
            matches = re.findall(r"\d{17}[\dXx]", txt)
            for id_num in matches:
                if self.verifyByIDCard(id_num):
                    self.res["idNumber"] = id_num
                    self.res["gender"] = "男" if int(id_num[16]) % 2 else "女"
                    return  # 成功提取后直接返回

    def full_name(self):
        """
        sfz姓名
        """
        # 如果姓名后面有跟文字,则取名后面的字段,如果"名"不存在,那肯定也就没有"姓名",所以在没有"名"的情况下只要判断是否有"姓"就可以了
        # 名字限制是2位以上,所以至少这个集合得3位数,才进行"名"或"姓"的判断

        for i, txt in enumerate(self.result):
            if ("姓名" in txt or "名" in txt or "姓" in txt) and len(txt) > 3:
                resM = re.findall(r"名[\u4e00-\u9fa5]+", txt)
                resX = re.findall(r"姓[\u4e00-\u9fa5]+", txt)
                if resM:
                    name = resM[0].split("名")[-1]
                elif resX:
                    name = resX[0].split("姓")[-1]
                else:
                    name = ""
                if len(name) > 1:
                    self.res["name"] = name
                    self.result[i] = "temp"  # 移除掉姓名字段,防止后面误识别
                    return

    def sex(self):
        """
        性别女民族汉
        """
        for txt in self.result:
            if "男" in txt:
                self.res["gender"] = "男"
            elif "女" in txt:
                self.res["gender"] = "女"

    def national(self):
        # 性别女民族汉
        # 先判断是否有"民族xx"或"族xx"或"民xx"这种类型的数据,有的话获取xx的数据,然后在56个名族的字典里判断是否包含某个民族,包含则取对应的民族
        keyword = ["男", "女", "性别", "民", "族", "民族"]
        for i, txt in enumerate(self.result):
            if any(k in txt for k in keyword):
                for nation in self.nation_list:
                    if nation in txt:
                        self.res["nationality"] = nation
                        self.result[i] = "temp"  # 移除掉民族字段,防止后面误识别
                        return

    def address(self):
        """
        地址
        """
        addString = []
        for i in range(len(self.result)):
            txt = self.result[i]
            # 这步的操作是去除下”公民身份号码“里的号对地址的干扰
            txt = txt.replace("号码", "")
            if "公民" in txt:
                txt = "temp"
            # sfz地址    盟,旗,苏木,嘎查  蒙语行政区划  ‘大学’有些大学集体户的地址会写某某大学

            if (
                "住址" in txt
                or "址" in txt
                or "省" in txt
                or "市" in txt
                or "县" in txt
                or "街" in txt
                or "乡" in txt
                or "村" in txt
                or "镇" in txt
                or "区" in txt
                or "城" in txt
                or "室" in txt
                or "组" in txt
                or "号" in txt
                or "栋" in txt
                or "巷" in txt
                or "盟" in txt
                or "旗" in txt
                or "苏木" in txt
                or "嘎查" in txt
                or "大学" in txt
            ):
                # 默认地址至少是在集合的第2位以后才会出现,避免经过上面的名字识别判断未能识别出名字,
                # 且名字含有以上的这些关键字照成被误以为是地址,默认地址的第一行的文字长度要大于7,只有取到了第一行的地址,才会继续往下取地址
                if i < 2 or len(addString) < 1 and len(txt) < 7:
                    continue
                    # 如果字段中含有"住址"、"省"、"址"则认为是地址的第一行,同时通过"址"
                # 这个字分割字符串
                if "住址" in txt or "省" in txt or "址" in txt:
                    # 通过"址"这个字分割字符串,取集合中的倒数第一个元素
                    addString.insert(0, txt.split("址")[-1])
                else:
                    addString.append(txt)
                self.result[i] = "temp"

        if len(addString) > 0:
            self.res["address"] = "".join(addString)
        else:
            self.res["address"] = ""

    def predict_name(self):
        """
        如果PaddleOCR返回的不是姓名xx连着的,则需要去猜测这个姓名
        """
        name_pattern = re.compile(r"[\u4e00-\u9fa5]{2,4}")
        for txt in self.result:
            if not self.res["name"]:
                if 1 < len(txt) < 5:
                    if all(
                        keyword not in txt
                        for keyword in [
                            "性别",
                            "姓名",
                            "民族",
                            "住址",
                            "出生",
                            "号码",
                            "身份",
                        ]
                    ):
                        result = name_pattern.findall(txt)
                        if result:
                            self.res["name"] = result[0]
                            break

    def run(self):
        self.full_name()
        self.sex()  # 通过文字识别性别
        self.national()
        self.birth_no()  # 通过sfz号码识别性别
        self.address()
        self.predict_name()
        return self.out


class IdCardFan:
    """
    sfzOCR返回结果,正常的sfz大概率识别没有什么问题,少数名族sfz,壮文、藏文、蒙文基本识别也没问题
    """

    def __init__(self, result):
        self.result = result
        self.result0 = deepcopy(result)
        # 排除 含有'共和国' 和'sfz' 的字段
        self.result = [i for i in self.result if "共和国" not in i and "sfz" not in i]

        self.out = {"result": {}}
        self.res = self.out["result"]
        self.res["issue"] = ""
        self.res["valid_begin"] = ""
        self.res["valid_end"] = ""

    def issue(self):
        """
        签发机关
        """
        ftext = ""
        keywords = ["公安局", "局", "公安", "派驻所", "派出所", "所", "市", "县", "区"]
        for txt in self.result:
            if any(keyword in txt and len(txt) > 3 for keyword in keywords):
                ftext = txt
                break

        # 批量替换需要移除的关键字
        removal_keys = ["签发", "机关", "签发机关", "签", "发", "机", "关"]
        for key in removal_keys:
            ftext = ftext.replace(key, "")

        self.res["issue"] = ftext

    def valid(self):
        """
        提取sfz有效期限（起始与结束日期或年份）
        """
        is_long = False
        for _, txt in enumerate(self.result):
            # 检查是否有长期字段,如果有,则把valid_end置为长期
            if "长期" in txt or re.search(r"\d{4,}[长期]", txt):
                self.res["valid_end"] = "长期"
                is_long = True

        if is_long:
            for _, txt in enumerate(self.result):
                # 因此只需要找8位数的日期
                txt = re.sub(r"[^\w]", "", txt)
                # 尝试提取8位数字（如：20170920）
                matches = re.findall(r"\d{8}", txt)
                if matches:
                    self.res["valid_begin"] = matches[0]
                    return
            # 如果遍历了所有的字段,都没找到8位数的日期,则尝试提取4位以上的数字(只提取年份)
            for _, txt in enumerate(self.result):
                # 尝试提取4位以上的数字（如：2017）
                years = re.findall(r"19\d{2}|20\d{2}", txt)
                if years:
                    self.res["valid_begin"] = years[0]
                    return

        else:
            for txt in self.result:
                # 清洗文本：移除标点符号、空格等非字母数字字符
                txt = re.sub(r"[^\w]", "", txt)
                # 尝试提取连续16位数字（如：2017092020270920）
                matches = re.findall(r"\d{16}", txt)
                if matches:
                    valid_str = matches[0]
                    begin = valid_str[:8]
                    end = valid_str[8:]
                    # 检查有效期是否合理
                    if not (
                        int(begin) < int(end) and (int(end) - int(begin)) % 10 == 0
                    ):
                        print(f"有效期不合法, begin:{begin}, end:{end}")
                    self.res["valid_begin"] = begin
                    self.res["valid_end"] = end
                    return
                # 如果没有16位数字,则尝试提取4位以上的数字(只提取年份)
                # 尝试从 1950-2099 年份中提取(有可能提取出错, 比如  '201709202027' 这种, 理论上是 2017, 2027, 但是实际上是 2017, 2020)
                ## 如何避免呢
                # 如果没有匹配到16位数字,则尝试提取年份（格式为 19xx 或 20xx）
                years = re.findall(r"19\d{2}|20\d{2}", txt)
                if (
                    len(years) == 2
                    and years[0] < years[1]
                    and (int(years[1]) - int(years[0])) % 10 == 0
                ):
                    self.res["valid_begin"] = years[0]
                    self.res["valid_end"] = years[1]
                    return

    def run(self):
        self.issue()
        self.valid()
        return self.out


class IdCardOCRIdentify:
    def __init__(self, ocr=None, usemodel="baiduonnx"):
        """
        ocr: ocr模型
        """
        assert usemodel in ["baiduonnx", "paddleocr", ""], (
            "usemodel must be baiduonnx or paddleocr or None"
        )
        self.usemodel = usemodel
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError:
            usemodel = "baiduonnx"
        if not ocr and not self.usemodel:
            self.ocr = PaddleOCR(
                use_angle_cls=True, lang="ch", show_log=False, debug=False
            )
            self.usemodel = "paddleocr"

        elif not ocr and self.usemodel == "baiduonnx":
            from imgocr import ImgOcr

            self.ocr = ImgOcr(use_gpu=False, is_efficiency_mode=True)
        else:
            # 其他的ocr模型
            raise ValueError("暂时不支持其他的ocr模型")
            # self.ocr = ocr
            # self.isonnx = None

    def ocrresult(self, img) -> list[str]:
        """
        返回识别结果
        """
        result = self.ocr.ocr(img)
        # 返回统一的格式, 只要文本,变成一个list
        if self.usemodel == "baiduonnx":
            # 不用处理,因为返回的以及是字典了
            pass
        elif self.usemodel == "paddleocr":
            result = self._convert_to_dict(result[0])
        txtArr = self._clean_text(result)
        return txtArr

    def _convert_to_dict(self, data) -> list:
        """
        将嵌套列表结构转换为字典列表形式。(针对百度的PaddleOCR返回的结果 )

        参数:
            data: 原始数据,包含多重嵌套的 box 和 (text, confidence) 元组。

        返回:
            list[dict]: 每个字典包含 'box', 'text', 'confidence' 三个键。
        """
        results = []
        for item in data:
            # 解除多余的嵌套
            box, (text, confidence) = item[0], item[1]
            results.append({"box": box, "text": text, "confidence": confidence})

        return results

    def _clean_text(self, result):
        txtArr = []
        for item in result:
            txt = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fa5]", "", item["text"])
            if "共和国" in txt or "身份证" in txt:
                continue
            txtArr.append(txt)
        return txtArr

    def _load_image(self, source):
        if isinstance(source, (str, Path)):
            img = cv2.imread(str(source))
        elif isinstance(source, np.ndarray):
            img = source
        else:
            raise ValueError("source must be a string or a numpy array")
        return img

    def extract_info_zheng(self, txtArr):
        return IdCardStraight(txtArr).run()

    def extract_info_fan(self, txtArr):
        return IdCardFan(txtArr).run()

    def _apply_mask(self, img, mask, fill_white=False):
        x1, y1, x2, y2 = map(int, mask)
        if fill_white:
            if img.ndim == 3:
                img[y1:y2, x1:x2, :3] = 255
            elif img.ndim == 2:
                img[y1:y2, x1:x2] = 255
        else:
            return img[int(1.2 * y2) :, :]
        return img

    def _crop_mask(self, img, cls, mask, cls_box=None, debug=True) -> tuple[Any, Any]:
        """根据mask裁剪图片,使得图片只返回需要识别的部分"""
        x1, y1, x2, y2 = map(int, mask)
        h, w = img.shape[:2]
        if cls == 0:
            # 反面
            if cls_box:
                xx1, yy1, xx2, yy2 = map(int, cls_box)
                # 设置裁剪范围（使用比例）
                cx = int((xx1 + xx2) / 2)
                cy = int((yy1 + yy2) / 2)
                crop_x1 = int(max(cx * 73 / 100, xx1, x2))
                crop_y1 = int(max(cy * 1.2, yy1, y2))

                crop_x1 = np.clip(crop_x1, 0, w)
                crop_y1 = np.clip(crop_y1, 0, h)
                crop_x_end = np.clip(int(xx2 * 0.9), 0, w)
                crop_y_end = np.clip(int(yy2 * 0.98), 0, h)
                img_sec1 = img[crop_y1:crop_y_end, crop_x1:crop_x_end]
            else:
                img_sec1 = img[y2:, x2:]

            if debug:
                cv2.imwrite("img_back.jpg", img_sec1)
            return img_sec1, None
        elif cls == 1:
            # 正面

            if cls_box:
                xx1, yy1, xx2, yy2 = map(int, cls_box)
                cx = int((xx1 + xx2) / 2)
                cy = int((yy1 + yy2) / 2)
                adj_x1 = max(int(cx * 0.35), xx1)
                adj_y1 = max(int(cy * 0.1), yy1)

                idx = int(cx * 0.6)
                # 防止越界
                adj_x1 = np.clip(adj_x1, 0, w)
                adj_y1 = np.clip(adj_y1, 0, h)
                idx = np.clip(idx, 0, w)

                img_sec1 = img[adj_y1:y2, adj_x1:x1]  # 不含sfz号码
                img_sec2 = img[y2:, idx:]  # 含sfz号码
            else:
                img_sec1 = img[:y2, :x1]  # 不含sfz号码
                img_sec2 = img[y2:, :]  # 含sfz号码
            if debug:
                cv2.imwrite("img_sec1.jpg", img_sec1)
                cv2.imwrite("img_sec2.jpg", img_sec2)
            return img_sec1, img_sec2
        # 如果没有匹配的cls,返回两个None,保证返回类型一致
        return None, None

    def _ocr_crop(self, img_sec1, img_sec2):
        """对裁剪后的图片进行OCR识别"""
        result = []
        if img_sec1 is not None:
            result_sec1 = self.ocrresult(img_sec1)
            result.extend(result_sec1)
        if img_sec2 is not None:
            result_sec2 = self.ocrresult(img_sec2)
            result.extend(result_sec2)
        return result

    def extract_info(
        self,
        img: str | np.ndarray,
        cls: int,
        mask: Optional[list] = None,
        cls_box=None,
        use_crop=True,
    ):
        """提取sfz信息

        Args:
            img (str | np.ndarray): 输入图片路径或图片
            cls (int): 1表示正面, 0表示反面, 必须写, 主要是为了更好的识别和提取信息
            mask (list): 用于国徽的mask, 可以不传,即一个国徽在图上的坐标, 左上, 右下
            cls_box (list): 用于sfz的box, 可以不传,即一个sfz在图上的坐标, 左上, 右下
            use_crop (bool): 是否使用裁剪后的图片进行识别, 默认是使用裁剪后的图片进行识别, 如果不传, 则默认使用原图进行识别,裁剪后信息可能少,但原图识别可能噪音多
        """
        img = self._load_image(img)
        assert cls in [0, 1], f"cls must be 0 or 1, but got {cls}"
        self.cls = cls  # 1表示正面, 0表示反面
        if not use_crop:
            if self.cls == 0 and mask:
                img = self._apply_mask(img, mask, fill_white=True)
            elif self.cls == 1 and mask:
                img = self._apply_mask(img, mask, fill_white=True)
            # 直接识别原图
            txtArr = self.ocrresult(img)
        else:
            img_sec1, img_sec2 = self._crop_mask(img, cls, mask, cls_box, debug=False)
            # 对裁剪后的图片进行OCR识别
            txtArr = self._ocr_crop(img_sec1, img_sec2)

        if not txtArr:
            raise ValueError("该图片没有识别到任何内容, 请检查图片是否清晰")

        if self.cls == 1:
            out = self.extract_info_zheng(txtArr)
        else:
            out = self.extract_info_fan(txtArr)

        out["result"]["cls"] = self.cls

        # 检查是否有没有识别到的信息(只对正面进行检查,因为反面在提取信息的时候以及处理过了)
        # 大多数只有名族识别不到
        # 特殊处理：正面且民族缺失
        if (
            cls == 1
            and out["result"]["idNumber"]
            and not out["result"]["nationality"]
            and mask
        ):
            img = img[:, : int(mask[0])]  # 裁剪头像区域
            res = self.ocrresult(img)
            restxt = [re.sub(r"[^\u4e00-\u9fa5]", "", line[1][0]) for line in res[0]]
            for txt in restxt:
                if "民" in txt or "族" in txt:
                    for nation in IdCardStraight.nation_list:
                        if any(i in txt for i in nation):
                            out["result"]["nationality"] = nation
                            break
        return out["result"]

    ############################################
    # 以下是一些辅助函数, 主要是为了处理图片的裁剪和旋转
    ############################################
    @staticmethod
    def _is_width_larger(coords: list[int]) -> bool:
        """计算矩形的宽度和高度, 如果宽度大于高度, 则返回 True, 否则返回 False

        Args:
            coords: 矩形的坐标, 是一个列表, 包含 4 个元素, 分别是 x1, y1, x2, y2, 表示左上角和右下角的坐标

        Returns:
            bool: 如果宽度大于高度, 则返回 True, 否则返回 False
        """
        x1, y1, x2, y2 = coords
        return (x2 - x1) >= (y2 - y1)

    def crop_image(
        self,
        img_path: str | Path,
        size: list,
    ) -> np.ndarray:
        """裁剪图片,根据给定的坐标, 返回裁剪后的图片(这个图片保证是宽矩形的)

        Args:
            img_path: 输入的图片, 可以是路径或者图片
            size: 裁剪的区域, [x1, y1, x2, y2], 表示裁剪的区域,左上角和右下角的坐标

        Returns:
            img2: 裁剪后的图片, 宽矩形的图片
        """
        # 0. 检查参数的合法性
        assert len(size) == 4, "size 参数必须是 4 个元素的列表"

        # 1. 读取图片
        img = self._load_image(img_path)

        # 2,获取图片的高度和宽度
        h, w = img.shape[:2]

        x1, y1, x2, y2 = map(int, size)
        # # 4. 对给定的大小进行扩展, 扩展 2% 的像素, 但是不能小于 1, 也不能大于 5
        expand_n = max(1, min(int(min(w, h) * 0.02), 5))
        x1, y1 = max(0, x1 - expand_n), max(0, y1 - expand_n)
        x2, y2 = min(w, x2 + expand_n), min(h, y2 + expand_n)
        # # 4. 对给定的大小进行收缩,收缩 2% 的像素, 但是不能小于 1, 也不能大于 5
        # shrink_n = max(1, min(int(min(w, h) * 0.1), 5))
        # x1, y1 = max(0, x1 + shrink_n), max(0, y1 + shrink_n)
        # x2, y2 = min(w, x2 - shrink_n), min(h, y2 - shrink_n)
        # 5. 裁剪
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        img_crop = img[y1:y2, x1:x2]
        # 5.1. 如果是不是宽矩形, 则旋转 90 度, 如何知道选择旋转 90 度还是 270 度呢?
        # 这里先旋转90度,把图片变成宽矩形, 但是不一定是正立的, 后续靠 is_need_rotate 来判断
        if not self._is_width_larger(size):
            img_crop = cv2.rotate(img_crop, cv2.ROTATE_90_CLOCKWISE)

        # 6. 调整大小,放大图片提高分辨率
        n = 4
        dstw = 240 * n  # 提高分辨率,原为240
        dsth = 151 * n  # 提高分辨率,原为151
        img_crop = cv2.resize(img_crop, (dstw, dsth), interpolation=cv2.INTER_CUBIC)

        return img_crop

    def is_need_rotate(
        self, img_path: str | Path | np.ndarray, class_id: int, size: list
    ) -> np.ndarray:
        """判断是否需要旋转180度,主要是根据头像和国徽的坐标来判断

        Args:
            img_path (str | Path | np.ndarray): 输入的图片, 可以是路径或者图片
            class_id: 目标类别, 0: fan, 1: zheng, 2: touxiang, 3: guohui
            size: 定义矩形坐标的四个整数的列表 [x1, y1, x2, y2]。

        Returns:
            img: 旋转后的图片
        """
        if class_id not in [2, 3]:
            raise ValueError("class_id 必须是 2（头像）或 3（国徽）")
        if len(size) != 4:
            raise ValueError("size 参数必须包含 4 个整数")
        img = self._load_image(img_path)
        # 0. 初步检查, 获取图片的高度和宽度, 必须是宽矩形, 宽度大于高度
        h, w = img.shape[:2]
        if h >= w:
            raise ValueError("图片应为宽矩形,宽度应大于高度")
        center_x = w // 2
        # center_y = h // 2

        # 1. 要求输入 touxiang 和 guohui 的坐标
        x1, y1, x2, y2 = map(int, size)
        assert x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h, "坐标超出范围"
        # 计算头像或国徽的中心坐标
        cx1, _cy1 = (x1 + x2) // 2, (y2 + y1) // 2
        if class_id == 2 and cx1 < center_x:
            # 如果头像的中心坐标在图片的左半部分, 需要旋转180度
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif class_id == 3 and cx1 > center_x:
            # 如果国徽的中心坐标在图片的右半部分, 需要旋转180度
            img = cv2.rotate(img, cv2.ROTATE_180)
        return img

    def draw_rectangle(
        self,
        img_path: Union[str, Path, np.ndarray],
        size: list,
        color: tuple[int, int, int] = (0, 255, 0),
        text: str = "",
        thickness: int = 2,
        font_scale: float = 1.0,
        text_color: Optional[tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """在图像上绘制矩形并可选地添加文字。

        这个函数可以用于在图像上绘制矩形框,并可选地在矩形框内添加文字。它支持多种输入格式,包括图像路径、numpy 数组等。

        Args:
            img_path (Union[str, Path, np.ndarray]): 输入图像的路径或 numpy 数组。
            size (list): 定义矩形坐标的四个整数的列表 [x1, y1, x2, y2]。即画的矩形的坐标
            color (tuple): 矩形颜色的 BGR 格式元组。默认为绿色 (0, 255, 0)。
            text (str): 矩形内部要绘制的可选文字。默认为空字符串。
            thickness (int): 矩形边框的厚度。默认值为 2。
            font_scale (float): 文字大小的缩放因子。默认值为 1.0。
            text_color (Optional[tuple]): 文字颜色的 BGR 格式。如果未指定,默认为矩形颜色。

        Returns:
            np.ndarray: 绘制了矩形（和可选文字）的图像。

        """
        # 0. 检查参数的合法性
        img = self._load_image(img_path)
        # 把 text 为数字的转换为字符串
        if isinstance(text, (int, float)):
            text = str(text)
        # 检查输入的大小是否为 4 个元素的list和元组
        assert len(size) == 4, "size 必须包含 4 个整数"

        x1, y1, x2, y2 = map(int, size)

        # 画矩形
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # 画文字
        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_color = text_color or color
            text_pos = (x1, y1 - 10) if y1 - 10 > 10 else (x1, y1 + 20)
            cv2.putText(
                img,
                text,
                text_pos,
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

        return img


class IDCardOCR:
    CLSMAP = {
        0: "fan",
        1: "zheng",
        2: "touxiang",
        3: "guohui",
    }

    def __init__(
        self,
        model: str | Path,
        temp_dir: str | Path | None = None,
    ):
        """
        初始化sfz处理器

        Args:
            model (Optional[str]): 模型路径
            temp_dir (Optional[Union[str, Path]]): 临时目录路径（默认为 None,使用系统默认临时目录）, 临时文件目录下的文件夹和文件,会被定期清理,默认是30天.
        """
        self.model = model
        self.det = Detector(model, imgsz=(640, 640))
        self.iocr = IdCardOCRIdentify()

        if temp_dir:
            self.temp_dir = Path(temp_dir)
        else:
            # tempfile.TemporaryDirectory() # 创建临时目录,完成上下文后会自动删除
            # tempfile.mkdtemp() 用户用完临时目录后需要自行将其删除, 返回新目录的绝对路径。
            self._tempdir = tempfile.mkdtemp()
            self.temp_dir = Path(self._tempdir)

    @staticmethod
    def _find_first(detections: list[dict], target_cls: list[int]) -> dict:
        """在检测结果中查找第一个匹配的目标"""
        for d in detections:
            if d["cls"] in target_cls:
                return d
        raise ValueError("未找到匹配的目标")

    def process_image(
        self,
        img_path: Union[str, Path],
    ) -> dict:
        """主流程入口：处理sfz图片,返回识别信息

        Args:
            img_path (str): 输入图片路径

        Returns:
            dict: 识别结果,包括sfz的上的文字信息,只提取重要的
        """
        img_path = Path(img_path)
        assert img_path.exists(), f"图片不存在: {img_path}"

        ## 创建临时目录
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: 初次检测：定位卡片区域
        det_results = self.det.detect(img_path)[0]

        card_side = self._find_first(det_results, [0, 1])
        card_symbol = self._find_first(det_results, [2, 3])

        if not card_side or not card_symbol:
            raise ValueError("没有检测到正反面或头像和国徽, 请检查图片是否清晰!!")

        # Step 2: 裁剪sfz的区域图像
        cropped_img = self.iocr.crop_image(img_path, size=card_side["box"])
        temp_img_path = self.temp_dir / "temp_image.jpg"

        cv2.imwrite(str(temp_img_path), cropped_img)

        # step 3: 旋转摆正, 检测裁剪图像并判断是否需旋转
        second_det_results = self.det.detect(str(temp_img_path))[0]
        # 小心坐标有负数和超出范围的值
        symbol = self._find_first(second_det_results, [2, 3])

        if not symbol:
            raise ValueError("未在裁剪图像中识别出头像或国徽")

        aligned_img = self.iocr.is_need_rotate(
            str(temp_img_path), symbol["cls"], symbol["box"]
        )
        # 保存旋转后的图像
        final_img_path = self.temp_dir / "cropped_image.jpg"
        cv2.imwrite(str(final_img_path), aligned_img)

        ### Step 4: 利用旋转后的图像做最终检测并识别
        final_dets = self.det.detect(str(final_img_path))[0]
        card_side = self._find_first(final_dets, [0, 1])
        card_symbol = self._find_first(final_dets, [2, 3])
        if not card_side or not card_symbol:
            raise ValueError("对齐后的图像无法识别sfz关键区域")

        # 这里的side是正或反面, 需要传入到iocr中
        side = card_side["cls"]
        # 这里的mask是国徽或头像的box, 需要传入到iocr中
        mask = card_symbol["box"]
        # 这里的cls_box是sfz的box, 需要传入到iocr中
        cls_box = card_side["box"]

        info = self.iocr.extract_info(
            str(final_img_path), side, mask, cls_box, use_crop=True
        )

        # 把其中的cls转为类别
        info["cls"] = self.CLSMAP.get(info["cls"], "unknown")
        # 清除临时目录
        self._clean_temp()
        return info

    def _clean_temp(self):
        """清理临时目录"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


__all__ = ["IDCardOCR"]
