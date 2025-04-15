import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import os
from glob import glob

# 处理每一个NAIP文件
def process_file(naip_file, mask_folder, albedo_data, naip_pervious_folder, pervious_albedo_folder):
    with rasterio.open(naip_file) as naip_src:
        naip_data = naip_src.read()
        naip_transform = naip_src.transform
        naip_profile = naip_src.profile

        # 创建输出的Albedo影像，与NAIP影像大小一致
        albedo_resampled = np.zeros((naip_src.height, naip_src.width), dtype=np.float32)

        # 重新投影Albedo数据到NAIP分辨率
        reproject(
            source=albedo_data,
            destination=albedo_resampled,
            src_transform=albedo_transform,
            dst_transform=naip_transform,
            src_crs=albedo_src.crs,
            dst_crs=naip_src.crs,
            resampling=Resampling.bilinear
        )

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


    # 将NAIP pervious mask应用于重新采样的Albedo数据
    pervious_albedo_label = albedo_resampled * mask_data

    # 保存结果为tif文件
    output_path = os.path.join(pervious_albedo_folder, os.path.basename(naip_file))
    output_profile = naip_profile.copy()
    output_profile.update({
        'count': 1,
        'dtype': 'float32'
    })
    with rasterio.open(output_path, 'w', **output_profile) as dst:
        dst.write(pervious_albedo_label, 1)


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
        'Philadelphia',
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
        # 'NewYorkCity'
    ]


for city in cities:

    naip_folder = f'/workspace/ericyi/Surface Albedo/data/{city}/1024'
    mask_folder = f'/workspace/ericyi/Surface Albedo/data/{city}/1024_Pervious_mask'

    albedo_dir = f'/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Albedo_10m'

    for file in os.listdir(albedo_dir):
        if file.startswith('Landsat7') and file.endswith('Albedo.tif'):
            file_name = file

    # file_name = 'Philadelphia_Albedo_10m_26916.tif'
    albedo_path = os.path.join(albedo_dir, file_name)

    naip_pervious_folder = f'/workspace/ericyi/Surface Albedo/data/{city}/10m/1024_Pervious/naip_pervious'
    pervious_albedo_folder = f'/workspace/ericyi/Surface Albedo/data/{city}/10m/1024_Pervious/pervious_albedo'

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
        output_path = os.path.join(pervious_albedo_folder, os.path.basename(naip_file))

        if os.path.exists(output_path):
            continue
        else:
            process_file(naip_file, mask_folder, albedo_data, naip_pervious_folder, pervious_albedo_folder)
            print(n, naip_file)
