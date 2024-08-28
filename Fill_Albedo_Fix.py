import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds


def fill_with_landsat_data(one_meter_tif_path, landsat_tif_path, output_path):
    """
    使用Landsat数据填充1米分辨率影像中的0值。

    参数:
    - one_meter_tif_path: 1米分辨率影像的路径。
    - landsat_tif_path: 30米分辨率Landsat影像的路径。
    - output_path: 输出影像的路径。
    """
    # 读取1米分辨率影像
    with rasterio.open(one_meter_tif_path) as one_meter_src:
        one_meter_data = one_meter_src.read(1)  # 假设我们只处理单波段数据
        one_meter_profile = one_meter_src.profile
        one_meter_bounds = one_meter_src.bounds

        # 从Landsat影像中裁剪出与1米影像相同范围的区域
        with rasterio.open(landsat_tif_path) as landsat_src:
            landsat_window = from_bounds(*one_meter_bounds, landsat_src.transform)
            landsat_data = landsat_src.read(1, window=landsat_window)
            landsat_data = np.pad(landsat_data, (
            (0, one_meter_data.shape[0] - landsat_data.shape[0]), (0, one_meter_data.shape[1] - landsat_data.shape[1])),
                                  'constant', constant_values=0)

            # 使用Landsat数据替换1米影像中的0值
            mask = (one_meter_data == 0)
            one_meter_data[mask] = landsat_data[mask]

    # 保存修改后的1米分辨率影像
    with rasterio.open(output_path, 'w', **one_meter_profile) as dst:
        dst.write(one_meter_data, 1)

city = 'Cleveland'

# 定义文件夹路径和Landsat影像路径
one_meter_folder = f'/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Albedo_UNet'  # 替换为1米分辨率影像的文件夹路径
landsat_tif_dir = f'/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Albedo_Landsat7'

for file in os.listdir(landsat_tif_dir):
    if file.startswith('Landsat7') and file.endswith('Albedo.tif'):
        file_name = file

landsat_tif_path = os.path.join(landsat_tif_dir, file_name)

output_folder = f'/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Albedo_UNet_Landsat7'  # 替换为输出文件夹的路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历1米分辨率影像文件夹中的所有tif影像
for filename in os.listdir(one_meter_folder):
    if filename.endswith('.tif'):
        one_meter_tif_path = os.path.join(one_meter_folder, filename)
        output_path = os.path.join(output_folder, filename)

        if os.path.exists(output_path):
            continue
        else:
            fill_with_landsat_data(one_meter_tif_path, landsat_tif_path, output_path)
            print(f"Processed {filename}")
