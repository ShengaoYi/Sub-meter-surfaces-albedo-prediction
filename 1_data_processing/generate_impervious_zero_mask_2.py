import rasterio
from rasterio.enums import Resampling
import numpy as np
import os

input_folder_path = r'/workspace/ericyi/Surface Albedo/data/LA/Impervious/NAIP_LA_2016_Impervious_512'
# 输出文件夹路径
output_folder_path = r'/workspace/ericyi/Surface Albedo/data/LA/Impervious/NAIP_LA_2016_Impervious_512_zero_mask'

# 确保输出文件夹存在
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

for filename in os.listdir(input_folder_path):
    if filename.endswith('.tif'):
        input_file_path = os.path.join(input_folder_path, filename)
        output_file_path = os.path.join(output_folder_path, filename)

        # 使用rasterio打开源文件
        with rasterio.open(input_file_path) as src:
            # 复制源文件的元数据用于输出文件
            out_meta = src.meta.copy()

            # 读取图像数据
            image_data = src.read(masked=True)  # 'masked=True' 会根据nodata值掩膜数据

            # 将nodata值替换为0
            if src.nodata is not None:
                image_data = image_data.filled(0)  # 将masked数组中的掩膜值替换为0

            # 更新元数据中的nodata值为None，因为我们不再使用它
            out_meta.update(nodata=None)

            # 使用修改后的元数据创建新的tif文件
            with rasterio.open(output_file_path, 'w', **out_meta) as dst:
                dst.write(image_data)

print("处理完成！")
