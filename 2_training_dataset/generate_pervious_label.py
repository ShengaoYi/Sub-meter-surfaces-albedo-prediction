import rasterio
import numpy as np
import os
from glob import glob


def process_file(naip_file, mask_folder, albedo_data, naip_pervious_folder, pervious_albedo_folder):
    
    with rasterio.open(naip_file) as naip_src:
        naip_data = naip_src.read()
        naip_transform = naip_src.transform
        naip_profile = naip_src.profile

    mask_file = os.path.join(mask_folder, os.path.basename(naip_file))

    with rasterio.open(mask_file) as mask_src:
        mask_data = mask_src.read(1)

    # 计算NAIP pervious图像
    naip_pervious = naip_data * mask_data

    # 输出NAIP pervious图像
    naip_pervious_output_path = os.path.join(naip_pervious_folder, os.path.basename(naip_file))
    naip_pervious_profile = naip_profile.copy()
    naip_pervious_profile.update(count=naip_pervious.shape[0])
    with rasterio.open(naip_pervious_output_path, 'w', **naip_pervious_profile) as dst:
        dst.write(naip_pervious)

    # 获取pervious像素的地理坐标
    pervious_positions = np.where(naip_pervious[0] != 0)
    x_coords, y_coords = naip_src.xy(pervious_positions[0], pervious_positions[1])

    # 转换坐标到Landsat数据的像素坐标
    rows_30m, cols_30m = zip(*[albedo_src.index(x, y) for x, y in zip(x_coords, y_coords)])
    rows_30m = np.array(rows_30m)
    cols_30m = np.array(cols_30m)

    # 确保坐标在Landsat数据范围内
    valid_positions = (rows_30m >= 0) & (rows_30m < albedo_data.shape[0]) & (cols_30m >= 0) & (
            cols_30m < albedo_data.shape[1])
    rows_30m_valid = rows_30m[valid_positions]
    cols_30m_valid = cols_30m[valid_positions]
    pervious_positions_valid = (pervious_positions[0][valid_positions], pervious_positions[1][valid_positions])

    # 创建pervious albedo label，初始全为0
    pervious_albedo_label = np.zeros_like(naip_pervious[0], dtype=np.float32)

    # 使用Landsat的pervious albedo值填充
    pervious_albedo_label[pervious_positions_valid] = albedo_data[rows_30m_valid, cols_30m_valid]

    output_profile = naip_profile.copy()
    output_profile.update({
        'count': 1,
        'dtype': 'float32'
    })

    # 保存结果为tif文件
    output_path = os.path.join(pervious_albedo_folder, os.path.basename(naip_file))
    with rasterio.open(output_path, 'w', **output_profile) as dst:
        dst.write(pervious_albedo_label, 1)



cities = ['Philadelphia']


for city in cities:

    naip_folder = f'/workspace/ericyi/Surface Albedo/data/{city}/1024'
    mask_folder = f'/workspace/ericyi/Surface Albedo/data/{city}/1024_Pervious_mask'
    albedo_path = f'/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Albedo_Landsat7/Landsat7_Philadelphia_2022_26918_Albedo.tif'
    naip_pervious_folder = f'/workspace/ericyi/Surface Albedo/data/{city}/1024_Pervious/naip_pervious'
    pervious_albedo_folder = f'/workspace/ericyi/Surface Albedo/data/{city}/1024_Pervious/pervious_albedo'

    if not os.path.exists(naip_pervious_folder):
        os.makedirs(naip_pervious_folder)

    if not os.path.exists(pervious_albedo_folder):
        os.makedirs(pervious_albedo_folder)

    with rasterio.open(albedo_path) as albedo_src:
        albedo_data = albedo_src.read(1)
        albedo_transform = albedo_src.transform
    n = 0
    for naip_file in glob(os.path.join(naip_folder, '*.tif')):
        n += 1
        process_file(naip_file, mask_folder, albedo_data, naip_pervious_folder, pervious_albedo_folder)
        print(n, naip_file)

