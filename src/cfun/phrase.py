"""获取短语的正确语序

主要提供两个功能

1. 根据短语,返回正确的语序

2. 根据指定的字符串顺序,把数据进行排序


原理：

假设有四个字符(乱序), 需要得到正确的排序(语序)

1. 首先对这个四个字符进行排列组合,

2. 对每一个排列组合进行判断, 判断这个排列是否出现在数据库中（也可以是用户自定义的数据库, 第一列是 word, 第二列是 count(计数)）

    2.1 如果出现, 则返回这个排列组合

    2.2. 如果没有, 则返回None

3. 如果没有找到, 则可能是传递的 word 书写错误, 尝试利用正则进行匹配（模糊匹配）

    3.1 从匹配到的结果中,选出最有可能的组合

    3.2 如果匹配到的结果只有一个,则直接返回

4. 实在没有,则返回随机组合

"""

import importlib.resources as pkg_resources
import itertools
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import List

import pandas as pd
from char_similar_z import CharSimilarity

# 确保 src/cfun/data/__init__.py 存在
from . import data


class Phrase:
    """获取短语的正确语序

    Attributes:
        basedb_path (Path): 数据库路径
        userdb (pd.DataFrame): 用户自定义的数据库
        basedb (pd.DataFrame): 基础数据库
        db (pd.DataFrame): 合并后的数据库
        corrector_type (str): 纠错器类型
        corrector (Any): 纠错器对象
        chasimilarity (CharSimilarity): 字符相似度计算对象,由 char_similar_z 包提供
    """

    basedb_path = pkg_resources.files(data).joinpath("frequency.csv")

    def __init__(
        self, userdb: str = "", corrector: str = "charsimilar", onlyuser: bool = False
    ):
        """初始化类

        Args:
            userdb (str): 用户自定义的数据库路径, 如果不传入,则使用默认的数据库,上传了则会把用户的数据库和默认的数据库进行合并(用户优先)
            corrector (str): 纠错器类型, 可选值为 "charsimilar" 或 "bert", 默认为 "charsimilar"
            onlyuser (bool): 是否只使用用户自定义的数据库, 默认为 False, 如果为 True,则不加载默认的数据库
        """
        if onlyuser:
            self.userdb = self._load_dataframe(userdb) if userdb else None
            self.basedb = None
            self.db = self.userdb
        else:
            self.userdb = self._load_dataframe(userdb) if userdb else None
            self.basedb = self._load_dataframe(Path(str(self.basedb_path)))
            self.db = self._merge_databases()

        if corrector in ["bert", "charsimilar"]:
            self.corrector_type = corrector
        else:
            self.corrector_type = False
        self.corrector = self._loadcorrector() if self.corrector_type else None
        self.chasimilarity = CharSimilarity()

    @staticmethod
    def _generate_permutations(fragment: str) -> List[str]:
        """生成字符串的所有排列组合

        Args:
            fragment (str): 字符串

        Returns:
            List[str]: 所有排列组合的列表
        """
        assert isinstance(fragment, str) and len(fragment) > 0, (
            "fragment must be a non-empty string"
        )
        chars = list(fragment)
        assert all(len(c) == 1 for c in chars), (
            "Each character must be a single letter."
        )
        return ["".join(p) for p in itertools.permutations(chars)]

    def _loadcorrector(self):
        """加载纠错器"""
        if self.corrector_type == "bert":
            from pycorrector import MacBertCorrector

            return MacBertCorrector("shibing624/macbert4csc-base-chinese")
        elif self.corrector_type == "charsimilar":
            charsimilar = CharSimilarity()

            return charsimilar

    @staticmethod
    def _load_dataframe(path: Path | str) -> pd.DataFrame:
        """加载数据
        Args:
            path (Path | str): 数据库路径

        Returns:
            pd.DataFrame: 数据库
        """
        if isinstance(path, str):
            path = Path(path)
        assert path.exists(), f"{path} does not exist"
        if path.suffix == ".csv":
            return pd.read_csv(
                str(path), encoding="utf-8", dtype={"word": str, "count": int}
            )
        elif path.suffix == ".parquet":
            return pd.read_parquet(str(path))
        else:
            raise ValueError(
                "Unsupported file type. Only .csv and .parquet are supported."
            )

    # 模糊匹配 --- 正则匹配
    @staticmethod
    def _generate_regex_patterns(s, n=1):
        assert isinstance(s, str), "s must be a string"
        assert isinstance(n, int), "n must be an integer"
        assert 1 <= n <= len(s), "n 必须在 1 和字符串长度之间"

        patterns = []
        # 获取所有可能的 n 个位置的组合
        indices_combinations = itertools.combinations(range(len(s)), n)
        # 对每个组合生成对应的正则表达式

        for indices in indices_combinations:
            # 创建一个列表,把原字符串变为可变的列表
            # 用 '.' 替换这些位置的字符
            temp = list(s)
            for index in indices:
                temp[index] = "."

            # 把列表重新合成字符串
            patterns.append("".join(temp))
        return patterns

    @staticmethod
    def _get_ordinal_string(fragment, target) -> str:
        """
        根据findstring中字符的顺序重新排列phrase中的字符
        如果findstring中有的字符在phrase中没有出现,则用phrase中剩余的字符随机替换
        :param fragment: 原始字符串（目标顺序）   eg: 收下留情
        :param target: 匹配到的字符串（目标顺序） eg:  手下留情
        :return: 重排后的字符串
        """
        assert len(fragment) == len(target), (
            "fragment and target must have the same length"
        )

        # 利用原始的字符串进行组合, 因为我们找的的字符串和 fragment 不一定相同
        p_chars = list(fragment)
        f_chars = list(target)
        # 如果 fragment 中有重复字符（例如 ["a","a","b","c"]）,使用集合会丢失这个信息,因此使用列表推导式
        residual = [c for c in p_chars if c not in f_chars]
        random.shuffle(residual)  # 预先打乱剩余字符以提高随机性
        # 构建结果字符串
        residual_iter = iter(residual)
        # 这里使用 next(residual_iter, "") 是为了避免 StopIteration 异常
        return "".join(
            char if char in p_chars else next(residual_iter, "") for char in f_chars
        )

    def _merge_databases(self) -> pd.DataFrame:
        """合并用户数据库和基础数据库"""
        if self.userdb is None:
            return self.basedb  # type: ignore
        # 合并用户数据库和基础数据库
        df_user = self.userdb.copy()
        df_base = self.basedb.copy()  # type: ignore
        df = pd.concat([df_user, df_base]).drop_duplicates("word", keep="last")
        return df.sort_values(by="count", ascending=False).reset_index(drop=True)

    # 获取数据库中的匹配(完全匹配)
    def _best_match_from_db(self, permutations: List[str]) -> str:
        """
        批量查找数据库中的匹配
        :param permutations: 字符串列表(所有的可能组合)
        :return: 字符串
        eg:
        permutations = ["首虾留情", "首留情虾", "虾首留情", "虾留情首", "留情首虾", "留情虾首"]

        """
        assert isinstance(permutations, list), "permutations must be a list"
        matched = self.db[self.db["word"].isin(permutations)]  # type: ignore
        if not matched.empty:
            # 如果找到了匹配的词, 则返回第一个(频率最高的)
            matched = matched.drop_duplicates(subset=["word"], keep="last")
            matched = matched.sort_values(by="count", ascending=False).reset_index(
                drop=True
            )
            # 取出第一个词
            return matched.iloc[0]["word"]
        return ""

    # 模糊匹配 --- 正则匹配
    def _best_match_from_regex(
        self, permutations: List[str], n: int = 1
    ) -> tuple[str, pd.DataFrame]:
        """
        对带有正则通配符的字符串组合进行模糊匹配
        :param permutations: 所有的可能组合(里面含有正则通配符)
        :param n: 替换成正则通配符 '.' 的字符数量
        :return: 匹配到最有可能的字符串, 以及 匹配到的所有可能组成的 DataFrame
        :rtype: tuple[str, pd.DataFrame]
        :example:
        permutations = ["首虾留情", "首留情虾", "虾首留情", "虾留情首", "留情首虾", "留情虾首"]
        n = 1
        """
        assert isinstance(permutations, list), "permutations must be a list"
        assert len(permutations) > 0, "permutations must not be empty"

        # 预先成所有正则模式 (自动去重, 原因采用 集合推导式)
        patterns = list(
            {
                pattern
                for combo in permutations
                for pattern in self._generate_regex_patterns(combo, n)
            }
        )
        ### 快
        regex_pattern = "|".join(f"^(?:{p})$" for p in patterns)
        matched_df = self.db[self.db["word"].str.fullmatch(regex_pattern, na=False)]  # type: ignore
        if not matched_df.empty:
            # 如果找到了匹配的词, 则返回第一个(频率最高的), 和所有匹配的 DataFrame
            matched_df = matched_df.drop_duplicates(subset=["word"], keep="last")
            result_df = matched_df.sort_values(by="count", ascending=False).reset_index(
                drop=True
            )
            return result_df.iloc[0]["word"], result_df
        return "", pd.DataFrame(columns=["word", "count"])

    def _bertcorrector(self, candidates: list[str]) -> tuple[bool, str, str]:
        """
        检查是否有错误
        :param candidates: 字符串列表 (想要检查的组合)
        :param fragment: 原始字符串 (无顺序)
        :return: (是否有错误, 错误的字符, 正确的字符)
        """
        if not self.corrector_type:
            return False, "", ""
        # 计算相似度,self.corrector的是类对象,因此需要调用函数
        corrected_list = self.corrector.correct_batch(candidates)  # type: ignore

        # 提取并统计错误组合
        ie_list = [
            (e, r)
            for item in corrected_list
            if "errors" in item and len(item["errors"]) == 1
            for e, r, _ in item["errors"]
        ]
        # 如果没有错误组合,返回 False
        if not ie_list:
            return False, "", ""
        # 找出最常见的第一个元素
        first_elements_count = Counter(e for e, _ in ie_list)
        most_common_first = first_elements_count.most_common(1)[0][0]

        # 筛选出所有第一个元素为 most_common_first 的元素
        filtered_ie_list = [x for x in ie_list if x[0] == most_common_first]

        # 找出最常见的第二个元素(在第一个的基础上)
        last_elements_count = Counter(r for _, r in filtered_ie_list)
        most_common_last = last_elements_count.most_common(1)[0][0]
        # 筛选出所有第二个元素为 most_common_last 的元素
        final_ie_list = [x for x in filtered_ie_list if x[1] == most_common_last]
        # 检查所有元素是否相同
        if final_ie_list and all(x == final_ie_list[0] for x in final_ie_list):
            return True, final_ie_list[0][0], final_ie_list[0][1]

        return False, "", ""  # 如果没有找到符合条件的组合,返回 False

    def _best_char_pairs(
        self, errors: str, rights: str
    ) -> list[tuple[str, str, float]]:
        """获取最相似的字符对,

        依次对errors中的每个字符与rights中的每个字符进行比较,计算相似度,返回最相似的字符对, 返回的长度是 errors 的长度
        因为有些字是重复的,所以这里进行完全匹配。

        Args:
            errors (str): 错误的字符
            rights (str): 正确的字符

        Returns:
            list[tuple[str, str, float]]: 最相似的字符对

        Example:
            ```python
            from cfun.phrase import Phrase
            p = Phrase()
            errors = "人上七下"
            rights = "上七下"
            print(p._best_char_pairs(errors, rights))
            [('人', '七', 0.2823), ('上', '上', 1.0), ('七', '七', 1.0), ('下', '下', 1.0)]

            errors = "人上七下"
            rights = "万人之上"
            print(p._best_char_pairs(errors, rights))
            [('人', '人', 1.0), ('上', '上', 1.0), ('七', '之', 0.5203), ('下', '万', 0.5224)]
            ```
        """
        pairs = []
        used_rights = set()
        for e in errors:
            best_sim, best_r = 0, None
            for r in rights:
                sim = self.chasimilarity.std_cal_sim(e, r, 4, "pinyin")
                if sim > best_sim:
                    best_sim, best_r = sim, r
            if best_r:
                pairs.append((e, best_r, best_sim))
                used_rights.add(best_r)
        return pairs

    def _charsimilarcorrector(self, match_df: pd.DataFrame, fragment: str) -> str:
        """字符相似度纠错器

        Args:
            match_df (pd.DataFrame): 匹配到的 DataFrame
            fragment (str): 原始字符串 (无顺序)

        Returns:
            str: 最相似的字符串

        """
        # from char_similar_z import CharSimilarity

        # "all"(字形:拼音:字义=1:1:1)  # "w2v"(字形:字义=1:1)  # "pinyin"(字形:拼音=1:1)  # "shape"(字形=1)
        # 计算两个字符串的相似度, 先找出不同的字符,然后计算相似度
        # cc = []  # 存储错误的字符和正确的字符, 以及候选字符串

        best_word = ""
        best_score = 0
        for _, row in match_df.iterrows():
            word = row["word"]
            # 计算相似度
            pairs = self._best_char_pairs(fragment, word)
            # 平均相似度
            avg_sim = sum(sim for _, _, sim in pairs) / len(pairs)
            if avg_sim > best_score:
                best_score = avg_sim
                best_word = word

        return best_word

    def get_yuxu(self, fragment: str, maxn: int = 1) -> tuple[str, str, bool]:
        """获取最可能的语序组合

        Args:
            fragment (str): 传入的字符串
            maxn (int): 用于模糊匹配的字符数量, 默认为 1,建议设置为 1 或 2比较低的值,这样速度快很多

        Returns:
            tuple[str, str, bool]: (原始组合, 数据库匹配组合, 是否完全匹配)

        Raises:
            AssertionError: 如果 fragment 不是字符串类型

        Example:
            ```python
            from cfun.phrase import Phrase
            p = Phrase()
            fragment = "下收留情"
            reword, matched, isright = p.get_yuxu(fragment)
            print(reword, matched, isright)
            收下留情 手下留情 False
            ```

        """
        assert isinstance(fragment, str), "fragment must be a string"
        assert len(fragment) >= maxn >= 0 and isinstance(maxn, int), (
            "maxn must be a non-negative integer and less than the length of fragment"
        )
        # 获取所有排列组合
        combinations = self._generate_permutations(fragment)

        # 获取数据库中的匹配
        matched = self._best_match_from_db(combinations)
        if not matched and maxn == 0:
            return "".join(fragment), "", False

        # 如果没有找到, 则可能是传递的 fragment 中存在错别字, 尝试利用正则进行匹配（模糊匹配）
        if matched == "":
            matched, match_df = self._best_match_from_regex(combinations, n=maxn)
            # 如果 match_df 长度为空或为1,则直接返回
            if match_df.empty:
                matched = ""
            elif len(match_df) == 1:
                # 说明匹配到了,直接返回
                pass
            else:
                # print(f"匹配到多个结果: {match_df}")
                # 对模糊匹配的结果进行处理,找出最合理的字符串
                candidates = [
                    self._get_ordinal_string(fragment, i) for i in match_df["word"]
                ]
                if self.corrector_type == "bert":
                    # 这里的思路： 先把匹配到的字符串与原始字符串进行顺序还原,然后让bert纠错器进行纠错,看是否有错误,统计错误多的,
                    has_err, wrong, right = self._bertcorrector(candidates)
                    # 如果有错误的,则改成正确的
                    if has_err:
                        raw = candidates[0].replace(wrong, right)
                        combinations2 = self._generate_permutations(raw)
                        corrected = self._best_match_from_db(combinations2)
                        matched = corrected if corrected else matched
                elif self.corrector_type == "charsimilar":
                    # 这里的思路： 直接对模糊匹配的结果进行处理,找出所有错误,正确的字符组合,然后计算相似度,返回最相似的字符串
                    res = self._charsimilarcorrector(match_df, fragment)
                    matched = res if res else matched

        if matched != "":
            # 根据找到的组合,还原最终结果
            reword = self._get_ordinal_string(fragment, matched)
            return reword, matched, reword == matched
            # 返回, reword: 原来的语序, matched: 数据库中的语序(纠正后的), isfind: 是否正确
        else:
            # 证明在数据库中的确没有, 那么我们应该从网页上找(不一定能找到,因为错别字), 这里就直接返回随机组合
            return "".join(fragment), "", False

    def phrase_cw(
        self, data: list[dict], cw: str, key: str = "name", check: bool = True
    ) -> list[dict]:
        """根据字符串 cw 的顺序对 data 中的字典进行排序。

        要求输入参数 data 中的每个元素都必须是包含指定的key字段(默认是 "name"):

        - 若启用检查（check=True）且 data 的长度小于 cw 的长度,则会抛出异常。
        - 若字符无法精确匹配,则会计算相似度进行最优匹配。

        Args:
            data (list[dict]): 要排序的字典列表,且每个字典中必须包含指定的 key 字段（默认是 "name"）。
            cw (str): 指定的字符顺序。
            key (str, optional): 要用于匹配的字段名。默认为 "name"。
            check (bool, optional): 是否启用长度检查和严格匹配。默认为 True。

        Returns:
            list[dict]: 按照 cw 顺序排序后的字典列表。

        Raises:
            AssertionError: 如果参数类型或结构不符合要求。
            ValueError: 在 check 为 True 时,若无法找到匹配字符,则抛出。

        Example:
            ```python
            from cfun.phrase import Phrase
            p = Phrase()
            data = [{"name": "之", "coordinates": [207, 115]},
                    {"name": "成", "coordinates": [158, 86]},
                    {"name": "人", "coordinates": [126, 44]}]
            cw = "成人之"
            print(p.phrase_cw(data, cw))
            [{"name": "成", "coordinates": [158, 86]},
            {"name": "人", "coordinates": [126, 44]},
            {"name": "之", "coordinates": [207, 115]}]
            ```
        """
        assert isinstance(data, list) and len(data) > 0, "data必须是list类型,且不能为空"
        assert all(isinstance(item, dict) and key in item for item in data), (
            "data中的每个元素必须是字典类型,且必须包含name字段"
        )
        assert isinstance(cw, str) and len(cw) > 0, "cw必须是str类型,且不能为空"
        if check:
            assert len(data) >= len(cw), "参数cw的长度不能大于data的长度"

        remaining = deepcopy(data)  # 深拷贝,避免修改原数据
        result = []

        for ch in cw:
            # 尝试精确匹配 （计算快）
            ### next表示从生成器中取出第一个匹配项,没有匹配到则返回None
            exact_match = next((item for item in remaining if item[key] == ch), None)
            if exact_match:
                result.append(exact_match)
                remaining.remove(exact_match)
                continue
            # 尝试相似度匹配（直接使用相似度计算,因为如果两个字符完全相同,则相似度为1,慢一些）
            best_match = None
            best_score = 0
            for item in remaining:
                score = self.chasimilarity.std_cal_sim(ch, item[key], 4, "pinyin")
                if score > best_score:
                    best_score = score
                    best_match = item

            if best_match:
                result.append(best_match)
                remaining.remove(best_match)
            elif check:
                raise ValueError(f"未找到与字符 '{ch}' 匹配的项")
            else:
                result.append({})

        return result


if __name__ == "__main__":
    p = Phrase(corrector="charsimilar")
    fragment = ["下收留情", "手下留情", "人七上下", "情首虾留", "将相王候"]

    for f in fragment:
        reword, matched, isright = p.get_yuxu(f, maxn=0)
        print(f"原始: {f}, reword: {reword}, matched: {matched}, isright: {isright}")

    print("--" * 30)
    for f in fragment:
        reword, matched, isright = p.get_yuxu(f, maxn=1)
        print(f"原始: {f}, reword: {reword}, matched: {matched}, isright: {isright}")

    print("--" * 30)
    for f in fragment:
        reword, matched, isright = p.get_yuxu(f, maxn=2)
        print(f"原始: {f}, reword: {reword}, matched: {matched}, isright: {isright}")
    ######## 开启纠错器的情况下 ########
    # 原始: 下收留情, reword: 下收留情, matched: , isright: False
    # 原始: 手下留情, reword: 手下留情, matched: 手下留情, isright: True
    # 原始: 人七上下, reword: 人七上下, matched: , isright: False
    # 原始: 情首虾留, reword: 情首虾留, matched: , isright: False
    # 原始: 将相王候, reword: 将相王候, matched: , isright: False
    # ------------------------------------------------------------
    # 原始: 下收留情, reword: 收下留情, matched: 手下留情, isright: False
    # 原始: 手下留情, reword: 手下留情, matched: 手下留情, isright: True
    # 原始: 人七上下, reword: 七上人下, matched: 七上八下, isright: False
    # 原始: 情首虾留, reword: 情首虾留, matched: , isright: False
    # 原始: 将相王候, reword: 王候将相, matched: 王侯将相, isright: False
    # ------------------------------------------------------------
    # 原始: 下收留情, reword: 收下留情, matched: 手下留情, isright: False
    # 原始: 手下留情, reword: 手下留情, matched: 手下留情, isright: True
    # 原始: 人七上下, reword: 七上人下, matched: 七上八下, isright: False
    # 原始: 情首虾留, reword: 首虾留情, matched: 手下留情, isright: False
    # 原始: 将相王候, reword: 王候将相, matched: 王侯将相, isright: False
