from cfun.json import jsonp2json, recursive_parse_json


def test_parse_json():
    jsonp1 = '__JSONP_XXX_1({"data": {"dt": "a123456","ac": {"a": 11,"b": 20}}});'
    jsonp2 = '__JSONP_XXX_1({"data": {"dt": "a1(234)56","ac": {"a": 11,"b": 20}}});'
    json_data1 = jsonp2json(jsonp1)
    json_data2 = jsonp2json(jsonp2)
    print(json_data1)
    print(json_data2)

    data = '{"key": "value", "list": "[1, 2, 3]", "nested": "{"key": "value"}"}'
    parsed_data = recursive_parse_json(data)
    print(parsed_data)

    assert isinstance(jsonp2json(jsonp1), dict)
    assert isinstance(jsonp2json(jsonp2), dict)
    assert jsonp2json(jsonp1) == {"data": {"dt": "a123456", "ac": {"a": 11, "b": 20}}}
