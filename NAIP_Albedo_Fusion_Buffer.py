import os
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from rasterio.features import geometry_mask
from shapely.geometry import box

def slice_tiff_and_extract_roofs(input_image_path, roof_shp_path, naip_output_folder, roof_output_folder,
                                 tile_size=512, pad_size=30):
    with rasterio.open(input_image_path) as src:
        roofs = gpd.read_file(roof_shp_path, encoding='utf-8').to_crs(src.crs)

        width, height = src.width, src.height
        meta = src.meta.copy()

        nodata = src.nodata

        num_tiles_x = int(np.ceil(width / tile_size))
        num_tiles_y = int(np.ceil(height / tile_size))

        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                # 定义原始tile的起始和结束点
                x_start, y_start = j * tile_size, i * tile_size
                x_end, y_end = min(x_start + tile_size, width), min(y_start + tile_size, height)

                # 定义原始Window来读取数据
                win = Window(col_off=x_start, row_off=y_start, width=x_end - x_start, height=y_end - y_start)
                data = src.read(window=win)

                # 检查是否所有像素为0或nodata
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

                # Extract roofs within the padded tile bounds
                clipped_roofs = roofs[roofs.geometry.intersects(box(*padded_win_bounds))]

                # Generate the roof albedo image for the current padded tile
                padded_shape = (padded_y_end - padded_y_start, padded_x_end - padded_x_start)
                roof_image = np.zeros(padded_shape, dtype=np.float32)
                for _, roof in clipped_roofs.iterrows():
                    calib_sr = roof['CALIB_SR']
                    roof_mask = geometry_mask([roof.geometry], out_shape=padded_shape, transform=padded_win_transform,
                                              invert=True)
                    roof_image[roof_mask] = calib_sr

                # Save the padded data and roof data
                padded_tile_meta = src.meta.copy()
                padded_tile_meta.update({
                    "height": padded_y_end - padded_y_start,
                    "width": padded_x_end - padded_x_start,
                    "transform": padded_win_transform
                })

                naip_tile_path = os.path.join(naip_output_folder, f"tile_{i}_{j}_padded.tif")
                roof_tile_path = os.path.join(roof_output_folder, f"tile_{i}_{j}_padded.tif")

                print(f"Padded NAIP tile saved: {naip_tile_path}")
                # print(f"Padded Roof tile saved: {roof_tile_path}")

                # Write data to file
                with rasterio.open(naip_tile_path, 'w', **padded_tile_meta) as dst:
                    dst.write(padded_data)
                with rasterio.open(roof_tile_path, 'w', driver='GTiff', height=roof_image.shape[0],
                                   width=roof_image.shape[1], count=1, dtype='float32', crs=src.crs,
                                   transform=padded_win_transform) as dst:
                    dst.write(roof_image, 1)


cities = ['LA', 'BA', 'LB', 'SF', 'SJ']

for city in cities:
    naip_roof_path = fr'/workspace/ericyi/Surface Albedo/data/{city}/NAIP/NAIP_{city}_Roof_2009_83.tif'
    roof_shp_path = fr'/workspace/ericyi/Surface Albedo/data/{city}/Albedo/{city}_Albedo_83.shp'
    naip_output_folder = fr'/workspace/ericyi/Surface Albedo/data/{city}/1024/naip_output'
    roof_output_folder = fr'/workspace/ericyi/Surface Albedo/data/{city}/1024/roof_output'

    if not os.path.exists(naip_output_folder):
        os.makedirs(naip_output_folder)
    if not os.path.exists(roof_output_folder):
        os.makedirs(roof_output_folder)

    # 调用示例
    slice_tiff_and_extract_roofs(naip_roof_path, roof_shp_path, naip_output_folder, roof_output_folder, pad_size=256)