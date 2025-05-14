from cfun.pan123 import Pan123openAPI


def test_pan123():
    import os

    access_token = os.environ.get("PAN123TOKEN")
    pan123 = Pan123openAPI()
    assert access_token is not None, "请设置环境变量 PAN"
    pan123.refresh_access_token(access_token)
    # # # 下载单个文件
    # fname = "epoch150.pt"
    # pan123.download(fname)
