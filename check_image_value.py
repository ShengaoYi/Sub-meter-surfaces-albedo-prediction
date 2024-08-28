import rasterio
import numpy as np
import os

# 文件夹路径
folder_path = '/workspace/ericyi/Surface Albedo/data/LA/512_RGB/naip_test'
n = 0
# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查是否为TIFF文件
    if filename.endswith('.tif') or filename.endswith('.tiff'):
        file_path = os.path.join(folder_path, filename)

        # 使用rasterio打开文件
        with rasterio.open(file_path, 'r+') as src:
            # 读取文件的所有波段数据
            data = src.read()

            # 检查是否为RGB图像（假设RGB图像有3个波段）
            if data.shape[0] == 3:
                # 将所有大于255的像素值设置为0
                data[data > 255] = 0

                # 将修改后的数据写回文件
                src.write(data)
    n += 1
    print(n)

print("Processing completed.")
