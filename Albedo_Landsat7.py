import rasterio
import os
import numpy as np


cities = [
    # 'Atlanta',
    # 'Austin',
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
    # 'NewYorkCity',
    'Phoenix'
    ]

for city in cities:

    input_folder = fr'/workspace/ericyi/Surface Albedo/data/{city}'
    files = os.listdir(input_folder)

    print(city)
    for file in files:
        if file.startswith('Landsat7') and file.endswith('.tif'):
            file_name = file


    input_raster = os.path.join(input_folder, file_name)

    output_folder = fr'/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Albedo_Landsat7'


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = os.path.join(output_folder, file_name.split('.')[0] + '_Albedo.tif')

    with rasterio.open(input_raster) as src:
        # Read the specific bands (1, 3, 4, 5, 7)
        blue = src.read(1)  # Assuming Band 1 is blue
        red = src.read(3)   # Assuming Band 3 is red
        nir = src.read(4)   # Assuming Band 4 is near-infrared
        swir1 = src.read(5) # Assuming Band 5 is shortwave infrared 1
        swir2 = src.read(6) # Assuming Band 7 is shortwave infrared 2

        # Calculate the broadband albedo
        albedo = (0.356 * blue + 0.130 * red + 0.373 * nir +
                  0.085 * swir1 + 0.072 * swir2 - 0.0018)

        albedo_clipped = np.clip(albedo, 0, 1)

        # Create a new dataset for albedo with the same profile as the source
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1)

        # Write the albedo raster
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(albedo_clipped.astype(rasterio.float32), 1)

