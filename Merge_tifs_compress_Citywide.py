import os
import numpy as np
import rasterio
from rasterio.merge import merge
import tempfile


def merge_tifs_with_rasterio(input_folder, output_file, batch_size=50):
    # 获取文件夹中的所有tif文件并按名称排序
    tif_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif')])

    # 按行和列的编号对tif文件进行排序
    tif_files_sorted = sorted(tif_files, key=lambda x: (int(x.split('_')[-2]), int(x.split('_')[-1].split('.')[0])))

    # 分批读取并合并tif文件，每个批次的结果存储为一个临时文件
    temp_files = []
    temp_dir = '/workspace/ericyi/Surface Albedo/data/tmp'

    for i in range(0, len(tif_files_sorted), batch_size):
        src_files_to_mosaic = [rasterio.open(fp) for fp in tif_files_sorted[i:i + batch_size]]
        mosaic, out_trans = merge(src_files_to_mosaic)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tif', dir=temp_dir).name
        temp_files.append(temp_file)

        with rasterio.open(temp_file, 'w', driver='GTiff', height=mosaic.shape[1], width=mosaic.shape[2],
                           count=mosaic.shape[0],
                           dtype=mosaic.dtype, crs=src_files_to_mosaic[0].crs, transform=out_trans) as dest:
            dest.write(mosaic)

        # 关闭当前批次的文件
        for src in src_files_to_mosaic:
            src.close()

    # 合并所有的临时mosaic
    src_files_to_final_mosaic = [rasterio.open(fp) for fp in temp_files]
    final_mosaic, out_trans = merge(src_files_to_final_mosaic)

    out_meta = src_files_to_final_mosaic[0].meta.copy()
    out_meta.update({
        'driver': 'GTiff',
        'height': final_mosaic.shape[1],
        'width': final_mosaic.shape[2],
        'transform': out_trans,
        'compress': 'lzw',  # Add LZW compression here
        'BIGTIFF': 'YES'
    })

    with rasterio.open(output_file, 'w', **out_meta) as dest:
        dest.write(final_mosaic)

    # 删除临时文件并关闭
    for fp, src in zip(temp_files, src_files_to_final_mosaic):
        os.remove(fp)
        src.close()

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
    input_folder = fr"/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Albedo_UNet_Masked_512_Full_new"
    output_file = fr"/workspace/ericyi/Surface Albedo/data/{city}/Albedo/{city}_Albedo_UNet.tif"
    merge_tifs_with_rasterio(input_folder, output_file, 100)
    print(city, 'done')
