import os
import numpy as np
import rasterio
from rasterio.windows import Window

def slice_single_tiff(input_image_path, output_folder, tile_size=512):
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

                # Save the tiles
                tile_meta = meta.copy()
                tile_meta.update({
                    "height": tile_size,
                    "width": tile_size,
                    "transform": rasterio.windows.transform(win, src.transform)
                })

                tile_path = os.path.join(output_folder, f"tile_{i}_{j}.tif")

                with rasterio.open(tile_path, 'w', **tile_meta) as dst:
                    dst.write(data)
# 'Memphis','Austin',
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
# 'LasVegas','Louisville',
# 'Miami',
# 'Milwaukee','Nashville',
# 'OklahomaCity',
# 'Philadelphia','LosAngeles',
# 'Minneapolis',
# 'Pittsburgh',
# 'Richmond',
# 'Sacramento',
# 'SaltLakeCity',
# 'SanAntonio',
# 'SanDiego','Seattle',
# 'StLouis',
# 'Houston',
# 'Phoenix',
# 'NewYorkCity'
cities = [

'SanFrancisco', 'Memphis'
]

root = r'/workspace/ericyi/Surface Albedo/data'

for city in cities:
    city_dir = os.path.join(root, city)

    NAIP_dir = os.path.join(city_dir, 'NAIP')

    files = os.listdir(NAIP_dir)

    if len(files) == 1:
        file = files[0]

        single_tiff_path = os.path.join(NAIP_dir, file)
        output_folder = os.path.join(city_dir, '512')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        print(f'Processing {city}')

        slice_single_tiff(single_tiff_path, output_folder)
    else:
        print(city)
