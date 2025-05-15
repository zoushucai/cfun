import os
from src.cfun.pan123 import Pan123openAPI

# 方式 1：已拥有 access_token
access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NDc4MDQwODYsImlhdCI6MTc0NzE5OTI4NiwiaWQiOjE4MTUyMDUwMDcsIm1haWwiOiIiLCJuaWNrbmFtZSI6Imx1Y2tseXpzYyIsInVzZXJuYW1lIjoxODIwODE2MTIwNywidiI6MH0.yQpB1xovxvWwzKEQlQeCAyVZTUTBUmIx7pvbXTkDS7c "
pan123 = Pan123openAPI()  # 创建对象
pan123.refresh_access_token(access_token)  # 设置 access_token



# 上传文件到根目录下（不覆盖，如有同名文件则会报错）
file_id = pan123.upload("gt3word_perfect_muticls_删除了一些自创的图片.rar", "gt3word_perfect_muticls_删除了一些自创的图片.rar", 0, True)
print(file_id)


# # 下载单个文件
# fname = "epoch150.pt"
# pan123.download(fname)

# # 下载多个文件
# filenames = ["epoch150.pt", "epoch200.pt"]
# pan123.download(filenames)