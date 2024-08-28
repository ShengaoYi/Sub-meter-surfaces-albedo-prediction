# -*- coding: utf-8 -*-
'''
@Time    : 2023/10/17 23:21
@Author  : Ericyi
@File    : unzip.py

'''
import os
import zipfile

# 指定文件夹路径
dir_path = r"F:\Ericyi\Surface Albedo\data\SJ\NAIP"
n = 0
# 遍历文件夹中的所有文件
for filename in os.listdir(dir_path):
    # 检查文件是否是.zip文件
    if filename.endswith(".ZIP"):
        # 获取.zip文件的完整路径
        zip_file_path = os.path.join(dir_path, filename)

        # 使用zipfile模块解压文件
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(dir_path)
        n += 1
        print(n)
        os.remove(zip_file_path)

print("All .zip files have been extracted!")
