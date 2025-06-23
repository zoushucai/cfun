"""与json相关的函数和类

该程序提供一些有关 json 的函数和类
暂时完成如下函数

Functions:
    jsonp2json: 将jsonp格式的字符串转换为dict格式
    recursive_parse_json: 递归解析json 字符串

"""

import json


def jsonp2json(obj: str) -> dict:
    """jsonp格式的字符串转换为dict格式

    将jsonp格式的字符串转换为dic格式,会删除掉jsonp的前缀和后缀

    Args:
        obj (str): jsonp格式的字符串

    Returns:
        dict: 利用json.loads()函数将字符串转换为dict格式

    Raises:
        ValueError: 输入的必须是字符串

    Example:
        ```python
        from cfun.json import jsonp2json
        jsonp1 = '__JSONP_XXX_1({"data": {"dt": "a123456","ac": {"a": 11,"b": 20}}});'
        jsonp2 = '__JSONP_XXX_1({"data": {"dt": "a1(234)56","ac": {"a": 11,"b": 20}}});'
        json_data1 = jsonp2json(jsonp1)
        json_data2 = jsonp2json(jsonp2)
        print(json_data1)
        {'data': {'dt': 'a123456', 'ac': {'a': 11, 'b': 20}}}

        print(json_data2)
        {'data': {'dt': 'a1(234)56', 'ac': {'a': 11, 'b': 20}}}

        ```
    """
    if not isinstance(obj, str):
        raise ValueError("Input must be a string.")
    start_idx = obj.find("(")
    end_idx = obj.rfind(")")
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        json_data = obj[start_idx + 1 : end_idx].strip()
        return json.loads(json_data)
    else:
        raise ValueError("No valid JSONP format found.")


def recursive_parse_json(data: str | dict | list) -> dict | list | str:
    """递归解析json 字符串

    主要是有些时候json字符串中又包含了dict和list类型的字符串.

    Args:
        data (str, dict, list): 输入数据,可以是字符串、字典或列表

    Returns:
        dict, list: 解析后的数据,字符串保持不变,字典和列表递归解析。

    Example:
        ```python

        from cfun.json import recursive_parse_json
        data = '{"key": "value", "list": "[1, 2, 3]", "nested": "{"key": "value"}"}'
        parsed_data = recursive_parse_json(data)
        print(parsed_data)
        {'key': 'value', 'list': [1, 2, 3], 'nested': {'key': 'value'}}

        ```
    """
    if isinstance(data, str):
        try:
            # 尝试解析字符串中的JSON对象
            parsed_data = json.loads(data)
            if isinstance(parsed_data, (dict, list)):
                return recursive_parse_json(parsed_data)
            return parsed_data
        except json.JSONDecodeError:
            return data
    elif isinstance(data, (dict, list)):
        if isinstance(data, dict):
            return {key: recursive_parse_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [recursive_parse_json(item) for item in data]
    return data


if __name__ == "__main__":
    jsonp1 = '__JSONP_XXX_1({"data": {"dt": "a123456","ac": {"a": 11,"b": 20}}});'
    jsonp2 = '__JSONP_XXX_1({"data": {"dt": "a1(234)56","ac": {"a": 11,"b": 20}}});'
    json_data1 = jsonp2json(jsonp1)
    json_data2 = jsonp2json(jsonp2)
    print(json_data1)
    print(json_data2)

    data = '{"key": "value", "list": "[1, 2, 3]", "nested": "{"key": "value"}"}'
    parsed_data = recursive_parse_json(data)
    print(parsed_data)
