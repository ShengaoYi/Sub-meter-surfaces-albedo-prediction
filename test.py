
import os
import rasterio


# def print_tif_shapes(folder_path):
#     # 遍历文件夹中的所有文件
#     for filename in os.listdir(folder_path):
#         # 检查文件扩展名是否为.tif
#         if filename.endswith(".tif"):
#             # 构建文件的完整路径
#             file_path = os.path.join(folder_path, filename)
#             # roof = os.path.join('/workspace/ericyi/Surface Albedo/data/LA/512/roof_train', filename)
#
#             # 使用rasterio打开.tif文件
#             with rasterio.open(file_path) as src:
#
#                 naip_image = src.read(out_dtype='float32')
#                 print(naip_image)
#             # with rasterio.open(roof) as src:
#             #
#             #     roof_image = src.read(out_dtype='float32')
#             #     print(roof_image)
#         break
#
# print_tif_shapes(r'/workspace/ericyi/Surface Albedo/data/Atlanta/512')

import os
# import numpy as np
# import rasterio
# from rasterio.merge import merge
# import tempfile
#
# def merge_tifs_with_rasterio(input_folder, output_file, batch_size=50):
#     # 获取文件夹中的所有tif文件并按名称排序
#     tif_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif')])
#
#     # 按行和列的编号对tif文件进行排序
#     tif_files_sorted = sorted(tif_files, key=lambda x: (int(x.split('_')[-2]), int(x.split('_')[-1].split('.')[0])))
#
#     # 分批读取并合并tif文件，每个批次的结果存储为一个临时文件
#     temp_files = []
#     for i in range(0, len(tif_files_sorted), batch_size):
#         src_files_to_mosaic = [rasterio.open(fp) for fp in tif_files_sorted[i:i + batch_size]]
#         mosaic, out_trans = merge(src_files_to_mosaic)
#
#         temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tif').name
#         temp_files.append(temp_file)
#
#         with rasterio.open(temp_file, 'w', driver='GTiff', height=mosaic.shape[1], width=mosaic.shape[2],
#                            count=mosaic.shape[0],
#                            dtype=mosaic.dtype, crs=src_files_to_mosaic[0].crs, transform=out_trans) as dest:
#             dest.write(mosaic)
#
#         # 关闭当前批次的文件
#         for src in src_files_to_mosaic:
#             src.close()
#
#     # 合并所有的临时mosaic
#     src_files_to_final_mosaic = [rasterio.open(fp) for fp in temp_files]
#     final_mosaic, out_trans = merge(src_files_to_final_mosaic)
#
#     with rasterio.open(output_file, 'w', driver='GTiff', height=final_mosaic.shape[1], width=final_mosaic.shape[2],
#                        count=final_mosaic.shape[0],
#                        dtype=final_mosaic.dtype, crs=src_files_to_final_mosaic[0].crs, transform=out_trans) as dest:
#         dest.write(final_mosaic)
#
#     # 删除临时文件并关闭
#     for fp, src in zip(temp_files, src_files_to_final_mosaic):
#         os.remove(fp)
#         src.close()
#
# city = "Boston"
#
# input_folder = fr"/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Albedo_UNet_Landsat7"
# output_file = fr"/workspace/ericyi/Surface Albedo/data/{city}/Albedo/{city}_Albedo.tif"
# merge_tifs_with_rasterio(input_folder, output_file, 50)
# import os
# import numpy as np
# import rasterio
# from rasterio.merge import merge
# import tempfile
#
#
# def merge_tifs_with_rasterio(input_folder, output_file, batch_size=50):
#     # 获取文件夹中的所有tif文件并按名称排序
#     tif_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif')])
#
#     # 按行和列的编号对tif文件进行排序
#     tif_files_sorted = sorted(tif_files, key=lambda x: (int(x.split('_')[-2]), int(x.split('_')[-1].split('.')[0])))
#
#     # 分批读取并合并tif文件，每个批次的结果存储为一个临时文件
#     temp_files = []
#     temp_dir = '/workspace/ericyi/Surface Albedo/data/tmp'
#
#     for i in range(0, len(tif_files_sorted), batch_size):
#         src_files_to_mosaic = [rasterio.open(fp) for fp in tif_files_sorted[i:i + batch_size]]
#         mosaic, out_trans = merge(src_files_to_mosaic)
#
#         temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tif', dir=temp_dir).name
#         temp_files.append(temp_file)
#
#         with rasterio.open(temp_file, 'w', driver='GTiff', height=mosaic.shape[1], width=mosaic.shape[2],
#                            count=mosaic.shape[0],
#                            dtype=mosaic.dtype, crs=src_files_to_mosaic[0].crs, transform=out_trans) as dest:
#             dest.write(mosaic)
#
#         # 关闭当前批次的文件
#         for src in src_files_to_mosaic:
#             src.close()
#
#     # 合并所有的临时mosaic
#     src_files_to_final_mosaic = [rasterio.open(fp) for fp in temp_files]
#     final_mosaic, out_trans = merge(src_files_to_final_mosaic)
#
#     out_meta = src_files_to_final_mosaic[0].meta.copy()
#     out_meta.update({
#         'driver': 'GTiff',
#         'height': final_mosaic.shape[1],
#         'width': final_mosaic.shape[2],
#         'transform': out_trans,
#         'compress': 'lzw',  # Add LZW compression here
#         'BIGTIFF': 'YES'
#     })
#
#     with rasterio.open(output_file, 'w', **out_meta) as dest:
#         dest.write(final_mosaic)
#
#     # 删除临时文件并关闭
#     for fp, src in zip(temp_files, src_files_to_final_mosaic):
#         os.remove(fp)
#         src.close()
#
#
# city = "NewYorkCity"
#
# input_folder = fr"/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Albedo_UNet_Masked_512_Full"
# output_file = fr"/workspace/ericyi/Surface Albedo/data/{city}/Albedo/{city}_Albedo_UNet.tif"
# merge_tifs_with_rasterio(input_folder, output_file, 200)
import os
import numpy as np
import rasterio
from rasterio.windows import Window

def slice_single_tiff(input_image_path, output_folder, tile_size=512, pad_size=256):
    with rasterio.open(input_image_path) as src:
        width, height = src.width, src.height
        meta = src.meta.copy()

        nodata = src.nodata

        num_tiles_x = int(np.ceil(width / tile_size))
        num_tiles_y = int(np.ceil(height / tile_size))

        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                x_start, y_start = j * tile_size, i * tile_size
                x_end, y_end = min(x_start + tile_size, width), min(y_start + tile_size, height)

                win = Window(col_off=x_start, row_off=y_start, width=x_end - x_start, height=y_end - y_start)
                data = src.read(window=win)

                # Skip tile if it's all black
                if nodata is not None:
                    if (data == 0).all() or (data == nodata).all():
                        continue
                else:
                    if (data == 0).all():
                        continue

                # Define the expanded window for reading
                padded_x_start = j * tile_size - pad_size
                padded_y_start = i * tile_size - pad_size
                padded_x_end = padded_x_start + tile_size + 2 * pad_size
                padded_y_end = padded_y_start + tile_size + 2 * pad_size

                # Define the expanded window
                padded_win = Window(col_off=padded_x_start, row_off=padded_y_start,
                                    width=padded_x_end - padded_x_start, height=padded_y_end - padded_y_start)

                # Read data using the expanded window, with padding
                padded_data = src.read(window=padded_win, boundless=True, fill_value=0)

                # Update the transform for the expanded window
                padded_win_transform = src.window_transform(padded_win)

                # Calculate bounds for clipping roofs
                padded_win_bounds = rasterio.windows.bounds(padded_win, src.transform)

                padded_tile_meta = src.meta.copy()
                padded_tile_meta.update({
                    "height": padded_y_end - padded_y_start,
                    "width": padded_x_end - padded_x_start,
                    "transform": padded_win_transform
                })

                tile_path = os.path.join(output_folder, f"tile_{i}_{j}.tif")
                print(tile_path)

                with rasterio.open(tile_path, 'w', **padded_tile_meta) as dst:
                    dst.write(padded_data)

# Example usage
single_tiff_path = r'/workspace/ericyi/Surface Albedo/data/Memphis/NAIP4/NAIP_Memphis_2021_26916.tif'
output_folder = r'/workspace/ericyi/Surface Albedo/data/Memphis/1024_new'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

slice_single_tiff(single_tiff_path, output_folder)
