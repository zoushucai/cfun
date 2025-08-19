import base64

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad


def AES_CBC(e: str, t: str) -> str:
    """
    模拟 JS 中的 CryptoJS.AES.encrypt (ECB + PKCS7)
    :param e: 明文
    :param t: 密钥
    :return: base64 编码的密文
    """
    key = t.encode("utf-8")
    data = e.encode("utf-8")

    cipher = AES.new(key, AES.MODE_ECB)  # ECB 模式
    encrypted = cipher.encrypt(pad(data, AES.block_size))  # PKCS7 填充
    return base64.b64encode(encrypted).decode("utf-8")
