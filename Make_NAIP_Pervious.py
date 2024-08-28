import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import os
from glob import glob

# 处理每一个NAIP文件
def process_file(naip_file, mask_folder, naip_pervious_folder):
    with rasterio.open(naip_file) as naip_src:
        naip_data = naip_src.read()
        naip_transform = naip_src.transform
        naip_profile = naip_src.profile


    mask_file = os.path.join(mask_folder, os.path.basename(naip_file))

    with rasterio.open(mask_file) as mask_src:
        mask_data = mask_src.read(1)

    naip_pervious = naip_data * mask_data

    # 输出NAIP pervious图像
    naip_pervious_output_path = os.path.join(naip_pervious_folder, os.path.basename(naip_file))
    naip_pervious_profile = naip_profile.copy()
    naip_pervious_profile.update(count=naip_pervious.shape[0])
    with rasterio.open(naip_pervious_output_path, 'w', **naip_pervious_profile) as dst:
        dst.write(naip_pervious)


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
        'Memphis',
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

    naip_folder = f'/workspace/ericyi/Surface Albedo/data/{city}/1024_new'
    mask_folder = f'/workspace/ericyi/Surface Albedo/data/{city}/1024_Pervious_mask_new'

    naip_pervious_folder = f'/workspace/ericyi/Surface Albedo/data/{city}/1024_Pervious/naip_pervious_new'

    if not os.path.exists(naip_pervious_folder):
        os.makedirs(naip_pervious_folder)

    n = 0
    for naip_file in glob(os.path.join(naip_folder, '*.tif')):
        n += 1

        process_file(naip_file, mask_folder, naip_pervious_folder)
        print(n, naip_file)
