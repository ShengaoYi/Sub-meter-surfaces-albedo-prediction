import os
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from rasterio.features import geometry_mask
from shapely.geometry import box

def slice_tiff_and_extract_roofs(input_image_path, roof_shp_path, naip_output_folder, roof_output_folder,
                                 tile_size=512):
    with rasterio.open(input_image_path) as src:
        roofs = gpd.read_file(roof_shp_path, encoding='utf-8').to_crs(src.crs)

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

                # 检查是否所有像素为0或nodata
                if nodata is not None:
                    if (data == 0).all() or (data == nodata).all():
                        continue
                else:
                    if (data == 0).all():
                        continue

                win_transform = rasterio.windows.transform(win, src.transform)

                # Extract roofs within the tile
                win_bounds = rasterio.windows.bounds(win, src.transform)
                clipped_roofs = roofs[roofs.geometry.intersects(box(*win_bounds))]

                # Generate the roof albedo image for the current tile
                roof_image = np.zeros((y_end - y_start, x_end - x_start), dtype=np.float32)
                for _, roof in clipped_roofs.iterrows():
                    calib_sr = roof['CALIB_SR']
                    roof_mask = geometry_mask([roof.geometry], out_shape=roof_image.shape, transform=win_transform,
                                              invert=True)
                    roof_image[roof_mask] = calib_sr

                # Save the tiles
                tile_meta = meta.copy()
                tile_meta.update({
                    "height": y_end - y_start,
                    "width": x_end - x_start,
                    "transform": win_transform
                })

                naip_tile_path = os.path.join(naip_output_folder, f"tile_{i}_{j}.tif")
                roof_tile_path = os.path.join(roof_output_folder, f"tile_{i}_{j}.tif")
                print(naip_tile_path)

                with rasterio.open(naip_tile_path, 'w', **tile_meta) as dst:
                    dst.write(data)
                with rasterio.open(roof_tile_path, 'w', driver='GTiff', height=roof_image.shape[0],
                                   width=roof_image.shape[1], count=1, dtype=np.float32, crs=src.crs,
                                   transform=win_transform) as dst:
                    dst.write(roof_image, 1)


# Example usage
naip_roof_path = r'/workspace/ericyi/Surface Albedo/data/LA/NAIP/NAIP_LA_Roof_2009_83.tif'
roof_shp_path = r'/workspace/ericyi/Surface Albedo/data/LA/Albedo/LA_Albedo_83.shp'
naip_output_folder = r'/workspace/ericyi/Surface Albedo/data/LA/512/naip_output'
roof_output_folder = r'/workspace/ericyi/Surface Albedo/data/LA/512/roof_output'

if not os.path.exists(naip_output_folder):
    os.makedirs(naip_output_folder)
if not os.path.exists(roof_output_folder):
    os.makedirs(roof_output_folder)

slice_tiff_and_extract_roofs(naip_roof_path, roof_shp_path, naip_output_folder, roof_output_folder)
