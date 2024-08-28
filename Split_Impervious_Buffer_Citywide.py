import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import from_bounds

def crop_impervious_areas(impervious_path, input_folder, impervious_folder, pervious_folder):
    """
    根据每个给定的 TIFF 文件的边界从 Impervious.tif 裁剪对应区域。

    :param impervious_path: 大的 Impervious TIFF 文件的路径。
    :param input_folder: 包含小 TIFF 文件的输入文件夹路径。
    :param output_folder: 输出裁剪 TIFF 文件的文件夹路径。
    """
    # 确保输出文件夹存在
    if not os.path.exists(impervious_folder):
        os.makedirs(impervious_folder)
    if not os.path.exists(pervious_folder):
        os.makedirs(pervious_folder)


    # 打开整个城市的不透水表面数据
    with rasterio.open(impervious_path) as impervious_src:

        # 遍历指定目录下的所有 TIFF 文件
        for tif_file in os.listdir(input_folder):
            tif_path = os.path.join(input_folder, tif_file)

            with rasterio.open(tif_path) as src:
                # 获取当前tif文件的边界
                bounds = src.bounds

                # 根据边界创建窗口
                window = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, impervious_src.transform)

                # 读取窗口对应的数据
                impervious_data = impervious_src.read(1, window=window)

                # 生成渗透数据，0变1，非0变0
                pervious_data = np.where(impervious_data == 0, 1, 0)

                # 设置输出文件的路径
                impervious_output_path = os.path.join(impervious_folder, tif_file)
                pervious_output_path = os.path.join(pervious_folder, tif_file)

                # 设置输出文件的元数据
                out_meta = impervious_src.meta.copy()
                out_meta.update({
                    'height': window.height,
                    'width': window.width,
                    'transform': rasterio.windows.transform(window, impervious_src.transform)
                })

                # 写入数据到新的 TIFF 文件
                with rasterio.open(impervious_output_path, 'w', **out_meta) as out_dst:
                    out_dst.write(impervious_data, 1)

                with rasterio.open(pervious_output_path, 'w', **out_meta) as out_dst:
                    out_dst.write(pervious_data, 1)

            print(tif_path)
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
    impervious_path = fr'/workspace/ericyi/Surface Albedo/data/{city}/NAIP_{city}_Impervious_new_padded.tif'
    input_folder = fr'/workspace/ericyi/Surface Albedo/data/{city}/1024_new'
    output_impervious_folder = fr'/workspace/ericyi/Surface Albedo/data/{city}/1024_Impervious_UNet_building_mask_new'
    output_pervious_folder = fr'/workspace/ericyi/Surface Albedo/data/{city}/1024_Pervious_mask_new'

    crop_impervious_areas(impervious_path, input_folder, output_impervious_folder, output_pervious_folder)

