import rasterio
import numpy as np
def single_to_three_band(single_band_path, three_band_path):
    with rasterio.open(single_band_path) as src:
        # Read the single band data
        single_band_data = src.read(1)

        # Ensure the single band data is read into a 2D numpy array
        if len(single_band_data.shape) == 2:
            # Create three identical bands from the single band data
            three_band_data = np.repeat(single_band_data[np.newaxis, :, :], 3, axis=0)
        else:
            raise ValueError("Unexpected data shape for single_band_data")

        # Update the metadata for the output file to reflect 3 bands
        out_meta = src.meta.copy()
        out_meta.update({
            'count': 3,  # Number of bands
            'dtype': 'uint8'  # Ensure data type is uint8, adjust if necessary
        })

        # Write the three-band data to the output file
        with rasterio.open(three_band_path, 'w', **out_meta) as dst:
            dst.write(three_band_data)

# Replace these paths with the paths to your files
single_band_path = r'E:\Project\Surface Albedo\data\LA\Impervious\Impervious_LA_2016_Single.tif'
three_band_path = r'E:\Project\Surface Albedo\data\LA\Impervious\Impervious_LA_2016_RGB.tif'

single_to_three_band(single_band_path, three_band_path)

# Note: This code assumes the input single-band raster is already in uint8 format.
# If your data is in a different format, you may need to adjust the 'dtype' in out_meta.update() accordingly.
