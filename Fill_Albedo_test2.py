# import os
#
# import rasterio
# import numpy as np
#
# cities = [
#     'Austin',
#     'DC',
#     'Dallas',
#     'Denver']
#
# for city in cities:
#
#     # 定义文件路径
#     high_res_folder = fr'/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Albedo_UNet'
#
#     landsat_tif_dir = f'/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Albedo_Landsat7'
#
#     for file in os.listdir(landsat_tif_dir):
#         if file.startswith('Landsat7') and file.endswith('Albedo.tif'):
#             file_name = file
#
#     low_res_fp = os.path.join(landsat_tif_dir, file_name)
#
#     output_folder = f'/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Albedo_UNet_Landsat7'
#
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # 打开低分辨率图像
#     with rasterio.open(low_res_fp) as src30m:
#         band30m = src30m.read(1)
#         profile30m = src30m.profile
#
#     n = 0
#     for high_res_fp in os.listdir(high_res_folder):
#         n += 1
#         new_filename = os.path.join(output_folder, high_res_fp)
#
#         if os.path.exists(new_filename):
#             continue
#
#         print(n, new_filename)
#
#         with rasterio.open(os.path.join(high_res_folder, high_res_fp)) as src1m:
#             band1m = src1m.read(1)
#             profile1m = src1m.profile
#
#         zero_positions = np.where(band1m == 0)
#
#         # 获取这些位置的地理坐标
#         x_coords, y_coords = src1m.xy(zero_positions[0], zero_positions[1])
#         try:
#             # 转换坐标到低分辨率图像的像素坐标
#             rows_30m, cols_30m = zip(*[src30m.index(x, y) for x, y in zip(x_coords, y_coords)])
#
#             # 将结果转换为numpy数组
#             rows_30m = np.array(rows_30m)
#             cols_30m = np.array(cols_30m)
#
#             # 确保坐标在低分辨率图像范围内
#             valid_positions = (rows_30m >= 0) & (rows_30m < band30m.shape[0]) & (cols_30m >= 0) & (cols_30m < band30m.shape[1])
#             rows_30m_valid = rows_30m[valid_positions]
#             cols_30m_valid = cols_30m[valid_positions]
#             zero_positions_valid = (zero_positions[0][valid_positions], zero_positions[1][valid_positions])
#
#             # 用低分辨率图像的值填充高分辨率图像的0值位置
#             band1m[zero_positions_valid] = band30m[rows_30m_valid, cols_30m_valid]
#
#             # 保存修改后的1m图像
#
#             with rasterio.open(new_filename, 'w', **profile1m) as dst:
#                 dst.write(band1m, 1)
#         except:
#             print('error')
#             continue
#
#
#

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
    for i in range(0, len(tif_files_sorted), batch_size):
        src_files_to_mosaic = [rasterio.open(fp) for fp in tif_files_sorted[i:i + batch_size]]
        mosaic, out_trans = merge(src_files_to_mosaic)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tif').name
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

    with rasterio.open(output_file, 'w', driver='GTiff', height=final_mosaic.shape[1], width=final_mosaic.shape[2],
                       count=final_mosaic.shape[0],
                       dtype=final_mosaic.dtype, crs=src_files_to_final_mosaic[0].crs, transform=out_trans) as dest:
        dest.write(final_mosaic)

    # 删除临时文件并关闭
    for fp, src in zip(temp_files, src_files_to_final_mosaic):
        os.remove(fp)
        src.close()

city = "Boston"

input_folder = fr"/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Albedo_UNet_Landsat7"
output_file = fr"/workspace/ericyi/Surface Albedo/data/{city}/Albedo/{city}_Albedo.tif"
merge_tifs_with_rasterio(input_folder, output_file, 50)
