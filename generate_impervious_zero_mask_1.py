import rasterio
from rasterio.enums import Resampling
import numpy as np
import os

# 原始文件夹路径
input_folder_path = '/workspace/ericyi/Surface Albedo/data/LA/Impervious/NAIP_LA_2016_Impervious_512'
# 输出文件夹路径
output_folder_path = '/workspace/ericyi/Surface Albedo/data/LA/Impervious/NAIP_LA_2016_Impervious_512_mask'

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

            # 读取所有波段的数据
            data = src.read()

            # 获取 nodata 值
            nodata = src.nodata

            mask = np.zeros(data.shape[1:], dtype=rasterio.uint8)

            # 将非nodata的像素设置为1
            for band in data:
                mask[band != nodata] = 1

            # 更新元数据以反映新的单波段掩膜
            out_meta.update(count=1, dtype=rasterio.uint8)

            # 使用修改后的元数据创建新的tif文件
            with rasterio.open(output_file_path, 'w', **out_meta) as dst:
                dst.write(mask, 1)

print("处理完成！")
