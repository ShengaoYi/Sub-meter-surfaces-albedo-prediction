import rasterio
from rasterio.enums import Resampling
import numpy as np
import os

# 原始文件夹路径
input_folder_path = 'E:/Project/Surface Albedo/data/LA/512/roof_train'
# 输出文件夹路径
output_folder_path = 'E:/Project/Surface Albedo/data/LA/512/mask_train'

# 确保输出文件夹存在
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 遍历文件夹中的所有tif文件
for filename in os.listdir(input_folder_path):
    if filename.endswith('.tif'):
        input_file_path = os.path.join(input_folder_path, filename)
        output_file_path = os.path.join(output_folder_path, filename)

        # 使用rasterio打开源文件
        with rasterio.open(input_file_path) as src:
            # 复制源文件的元数据用于输出文件
            out_meta = src.meta.copy()

            # 读取第一个波段的数据
            band1 = src.read(1)

            # 将不为0的像素值改为1
            band1[band1 != 0] = 1

            # 使用修改后的元数据创建新的tif文件
            with rasterio.open(output_file_path, 'w', **out_meta) as dst:
                dst.write(band1, 1)

print("处理完成！")

