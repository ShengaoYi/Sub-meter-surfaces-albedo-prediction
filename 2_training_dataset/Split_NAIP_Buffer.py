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
single_tiff_path = r'/workspace/ericyi/Surface Albedo/data/NewYorkCity/NAIP4/NAIP_NewYorkCity_2022_26918.tif'
output_folder = r'/workspace/ericyi/Surface Albedo/data/NewYorkCity/1024_new'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

slice_single_tiff(single_tiff_path, output_folder)
