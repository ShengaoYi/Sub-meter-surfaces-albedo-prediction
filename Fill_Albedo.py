import os
import shutil
import rasterio
import numpy as np

cities = [
    # 'Baltimore',
    # 'Boston',
    # 'Charlotte',
    # 'Chicago',
    # 'Cleveland'
    'NewYorkCity']

for city in cities:

    # 定义文件路径
    high_res_folder = fr'/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Albedo_UNet'

    landsat_tif_dir = f'/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Albedo_Landsat7'

    for file in os.listdir(landsat_tif_dir):
        if file.startswith('Landsat7') and file.endswith('Albedo.tif'):
            file_name = file

    low_res_fp = os.path.join(landsat_tif_dir, file_name)

    output_folder = f'/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Albedo_UNet_Landsat7'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开低分辨率图像
    with rasterio.open(low_res_fp) as src30m:
        band30m = src30m.read(1)
        profile30m = src30m.profile

    n = 0
    for high_res_fp in os.listdir(high_res_folder):
        n += 1
        new_filename = os.path.join(output_folder, high_res_fp)

        if os.path.exists(new_filename):
            continue

        print(n, new_filename)

        with rasterio.open(os.path.join(high_res_folder, high_res_fp)) as src1m:
            band1m = src1m.read(1)
            profile1m = src1m.profile

        zero_positions = np.where(band1m == 0)

        if zero_positions[0].size > 0 and zero_positions[1].size > 0:

            # 获取这些位置的地理坐标
            x_coords, y_coords = src1m.xy(zero_positions[0], zero_positions[1])
            # try:
                # 转换坐标到低分辨率图像的像素坐标
            rows_30m, cols_30m = zip(*[src30m.index(x, y) for x, y in zip(x_coords, y_coords)])

            # 将结果转换为numpy数组
            rows_30m = np.array(rows_30m)
            cols_30m = np.array(cols_30m)

            # 确保坐标在低分辨率图像范围内
            valid_positions = (rows_30m >= 0) & (rows_30m < band30m.shape[0]) & (cols_30m >= 0) & (cols_30m < band30m.shape[1])
            rows_30m_valid = rows_30m[valid_positions]
            cols_30m_valid = cols_30m[valid_positions]
            zero_positions_valid = (zero_positions[0][valid_positions], zero_positions[1][valid_positions])

            # 用低分辨率图像的值填充高分辨率图像的0值位置
            band1m[zero_positions_valid] = band30m[rows_30m_valid, cols_30m_valid]

            # 保存修改后的1m图像

            with rasterio.open(new_filename, 'w', **profile1m) as dst:
                dst.write(band1m, 1)
        else:
            shutil.copy(os.path.join(high_res_folder, high_res_fp), new_filename)
        # except:
        #     print('error')
        #     continue



