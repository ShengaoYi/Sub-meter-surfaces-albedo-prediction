import rasterio
import numpy as np
from rasterio.windows import Window
from rasterio.enums import Resampling

def pad_tiff(input_tif, pad_size, output_tif):
    """
    Pads a given TIFF file with zeros around its borders.

    Parameters:
    input_tif (str): Path to the input TIFF file.
    pad_size (int): Number of pixels to pad around each side of the image.
    output_tif (str): Path where the padded TIFF will be saved.
    """
    # Open the input TIFF file
    with rasterio.open(input_tif) as src:
        # Read the data from the file
        data = src.read(1)  # Assuming you're working with a single band

        # Create a new array with padding
        padded_data = np.pad(data, pad_width=pad_size, mode='constant', constant_values=0)

        # Update the metadata to reflect the new dimensions and transformation
        new_transform = src.transform * src.transform.translation(-pad_size, -pad_size)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": padded_data.shape[0],
            "width": padded_data.shape[1],
            "transform": new_transform,
            'compress': 'lzw',
            'BIGTIFF': 'YES'
        })

        # Write the padded data to a new TIFF file
        with rasterio.open(output_tif, 'w', **out_meta) as dest:
            dest.write(padded_data, 1)  # Write padded_data as band 1

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

    input_tif_path = f'/workspace/ericyi/Surface Albedo/data/{city}/NAIP_{city}_Impervious_new.tif'
    output_tif_path = f'/workspace/ericyi/Surface Albedo/data/{city}/NAIP_{city}_Impervious_new_padded.tif'
    pad_size = 256  # Example pad size

    pad_tiff(input_tif_path, pad_size, output_tif_path)

    print(city)
