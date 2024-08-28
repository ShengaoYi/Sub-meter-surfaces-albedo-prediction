import os
import rasterio

def convert_to_rgb(input_folder, output_folder):
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the input raster file
            with rasterio.open(input_path) as src:
                # Read the first three bands (RGB) from the source file
                rgb_data = src.read([1, 2, 3])

                # Define the metadata for the output file
                out_meta = src.meta.copy()
                out_meta.update({
                    'count': 3  # Number of bands
                })

                # Write the RGB data to the output file
                with rasterio.open(output_path, 'w', **out_meta) as dst:
                    dst.write(rgb_data)


# Replace 'input_folder_path' and 'output_folder_path' with your actual folder paths
input_folder_path = r'/workspace/ericyi/Surface Albedo/data/SF/512/naip_output'
output_folder_path = r'/workspace/ericyi/Surface Albedo/data/SF/512/naip_output_RGB'

convert_to_rgb(input_folder_path, output_folder_path)

