from cfun.freq import FREQUENCY


def test_frequency():
    print(FREQUENCY.head())
    print(FREQUENCY.columns)
    print(len(FREQUENCY))
    assert len(FREQUENCY) > 0, "FREQUENCY 数据为空"
    assert "word" in FREQUENCY.columns, "FREQUENCY 缺少 'word' 列"
    assert "count" in FREQUENCY.columns, "FREQUENCY 缺少 'count' 列"
