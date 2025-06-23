import time

from cfun.phrase import Phrase


def test_phrase():
    ### 涉及到torch的安装,以及模型的下载,比较麻烦,按需开启
    p = Phrase(corrector="charsimilar")
    fragment = ["下收留情", "手下留情", "人七上下", "情首虾留", "将相王候"]

    for f in fragment:
        reword, matched, isright = p.get_yuxu(f, maxn=0)
        print(f"原始: {f}, reword: {reword}, matched: {matched}, isright: {isright}")

    output = ["手下留情", "手下留情", "七上八下", "", "王侯将相"]
    print("--" * 30)
    for idx, f in enumerate(fragment):
        start = time.time()
        reword, matched, isright = p.get_yuxu(f, maxn=1)
        end = time.time()
        print(
            f"原始: {f}, reword: {reword}, matched: {matched}, isright: {isright}, time: {end - start:.4f}s"
        )
        assert matched == output[idx], f"测试失败: {f} -> {reword}"

    print("--" * 30)
    for f in fragment:
        reword, matched, isright = p.get_yuxu(f, maxn=2)
        print(f"原始: {f}, reword: {reword}, matched: {matched}, isright: {isright}")


def test_phrase_bert():
    p2 = Phrase(corrector="bert")
    fragment = ["下收留情", "手下留情", "人七上下", "情首虾留", "将相王候"]

    stime = time.time()
    for f in fragment:
        reword, matched, isright = p2.get_yuxu(f)
        print(f"原始: {f}, reword2: {reword}, matched2: {matched}, isright2: {isright}")
    print(f"bert time: {time.time() - stime:.4f}s")


def test_phrase_cw():
    result = [
        {
            "name": "正",
            "coordinates": [87, 90],
            "points": [[60, 59], [101, 59], [101, 102], [60, 102]],
            "image_width": 300,
            "image_height": 150,
            "rawimgmd5": "f12c826860200e599d0ce8c89f4b10fc",
        },
        {
            "name": "扣",
            "coordinates": [157, 72],
            "points": [[128, 42], [168, 42], [168, 86], [128, 86]],
            "image_width": 300,
            "image_height": 150,
            "rawimgmd5": "f12c826860200e599d0ce8c89f4b10fc",
        },
        {
            "name": "听",
            "coordinates": [50, 39],
            "points": [[24, 22], [65, 22], [65, 68], [24, 68]],
            "image_width": 300,
            "image_height": 150,
            "rawimgmd5": "f12c826860200e599d0ce8c89f4b10fc",
        },
        {
            "name": "埋",
            "coordinates": [27, 112],
            "points": [[9, 89], [49, 89], [49, 125], [9, 125]],
            "image_width": 300,
            "image_height": 150,
            "rawimgmd5": "f12c826860200e599d0ce8c89f4b10fc",
        },
        {
            "name": "梨",
            "coordinates": [252, 66],
            "points": [[222, 49], [264, 49], [264, 97], [222, 97]],
            "image_width": 300,
            "image_height": 150,
            "rawimgmd5": "f12c826860200e599d0ce8c89f4b10fc",
        },
    ]

    p = Phrase(corrector="charsimilar")

    data = p.phrase_cw(result, "听政扣梨")
    print(data)
