import rasterio
import numpy as np
import os
from glob import glob



# 函数处理不同类型的mask文件
def adjust_mask(mask_file, fill_value):
    with rasterio.open(mask_file) as mask_src:
        mask_data = mask_src.read(1)
        mask_profile = mask_src.profile

    adjusted_mask = np.full((1024, 1024), fill_value, dtype=mask_data.dtype)
    min_dim = min(mask_data.shape[0], 1024), min(mask_data.shape[1], 1024)
    adjusted_mask[:min_dim[0], :min_dim[1]] = mask_data[:min_dim[0], :min_dim[1]]

    # 保存调整后的mask文件
    mask_profile.update({'width': 1024, 'height': 1024})
    with rasterio.open(mask_file, 'w', **mask_profile) as dst:
        dst.write(adjusted_mask, 1)
    # print(f'Adjusted mask file saved: {mask_file}')


cities = [
        # 'Austin',
        # 'Atlanta',
        # 'Baltimore',
        # 'Boston',
        # 'Charlotte',
        # 'Chicago',
        # 'Cleveland',
        # 'DC',
        # 'Dallas',
        # 'Denver',
        # 'Detroit',
        # 'Indianapolis',
        # 'LasVegas',
        # 'Louisville',
        # 'Memphis',
        # 'Miami',
        # 'Milwaukee',
        # 'Nashville',
        # 'OklahomaCity',
        # 'Philadelphia',
        # 'LosAngeles',
        # 'Minneapolis',
        # 'Pittsburgh',
        # 'Richmond',
        # 'Sacramento',
        # 'SaltLakeCity',
        # 'SanAntonio',
        # 'SanDiego',
        # 'SanFrancisco',
        # 'Seattle',
        # 'StLouis',
        # 'Houston',
        # 'Phoenix',
        'NewYorkCity'
    ]


for city in cities:
    # 设置文件夹路径
    naip_folder = f'/workspace/ericyi/Surface Albedo/data/{city}/1024_new'
    pervious_mask_folder = f'/workspace/ericyi/Surface Albedo/data/{city}/1024_Pervious_mask_new'
    impervious_mask_folder = f'/workspace/ericyi/Surface Albedo/data/{city}/1024_Impervious_UNet_building_mask_new'

    # 获取NAIP文件列表
    pervious_files = glob(os.path.join(pervious_mask_folder, '*.tif'))

    n = 0

    for pervious_file in pervious_files:
        naip_name = os.path.basename(pervious_file)
        impervious_mask_file = os.path.join(impervious_mask_folder, naip_name)

        with rasterio.open(pervious_file) as mask_src:
            mask_data = mask_src.read(1)
            mask_profile = mask_src.profile

        if mask_data.shape[0] != 1024 or mask_data.shape[1] != 1024:
            n += 1
            adjust_mask(pervious_file, 1)
            adjust_mask(impervious_mask_file, 0)
            print(n, pervious_file)
