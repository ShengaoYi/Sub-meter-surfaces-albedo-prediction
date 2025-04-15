import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import os
from shapely.geometry import box
import numpy as np
import csv


def calculate_ratio(naip_folder, mask_folder):
    impervious_ratios = []
    pervious_ratios = []

    impervious_area = 0
    pervious_area = 0

    # Loop through NAIP tiles
    for naip_filename in os.listdir(naip_folder):
        if naip_filename.endswith(".tif"):
            naip_path = os.path.join(naip_folder, naip_filename)
            mask_filename = "impervious_" + naip_filename
            mask_path = os.path.join(mask_folder, mask_filename)

            # Read the NAIP tile
            with rasterio.open(naip_path) as naip_src:
                naip_tile = naip_src.read()
                # Count non-empty pixels (assuming 0 is the value for empty pixels)
                non_empty_pixels = np.count_nonzero(naip_tile, axis=0)
                non_empty_count = np.count_nonzero(non_empty_pixels)

            # Read the corresponding impervious mask
            with rasterio.open(mask_path) as mask_src:
                impervious_tile = mask_src.read(1)
                # Count the pixels with value 1 (impervious)
                impervious_count = np.count_nonzero(impervious_tile)

            impervious_area += impervious_count
            pervious_area += non_empty_count - impervious_count

            # Calculate the impervious ratio
            impervious_ratio = (impervious_count / non_empty_count) * 100 if non_empty_count else 0
            # Calculate the pervious ratio
            pervious_ratio = 100 - impervious_ratio

            impervious_ratios.append(impervious_ratio)
            pervious_ratios.append(pervious_ratio)

    impervious_area = impervious_area / 1e6
    pervious_area = pervious_area / 1e6

    return np.mean(pervious_ratios), np.mean(impervious_ratios), pervious_area, impervious_area, (pervious_area + impervious_area)


cities = [
    'Austin',
    'Atlanta',
    'Baltimore',
    'Boston',
    'Charlotte',
    'Chicago',
    'Cleveland',
    'DC',
    'Dallas',
    'Denver',
    'Detroit',
    'Indianapolis',
    'LasVegas',
    'Louisville',
    'Memphis',
    'Miami',
    'Milwaukee',
    'Nashville',
    'OklahomaCity',
    'Philadelphia',
    'LosAngeles',
    'Minneapolis',
    'Pittsburgh',
    'Richmond',
    'Sacramento',
    'SaltLakeCity',
    'SanAntonio',
    'SanDiego',
    'SanFrancisco',
    'Seattle',
    'StLouis',
    'Houston',
    'Phoenix',
    'NewYorkCity']

output_csv_path = '/workspace/ericyi/Surface Albedo/data/cities_impervious.csv'  # Change to your desired output path

with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(['City', 'Pervious_ratio', 'Impervious_ratio', 'Pervious_area', 'Impervious_area', 'Total_area'])

    for city in cities:

        NAIP_folder_path = f'/workspace/ericyi/Surface Albedo/data/{city}/512'
        tif_folder_path = f'/workspace/ericyi/Surface Albedo/data/{city}/512_Impervious_UNet_building_mask'
        pervious_ratio, impervious_ratio, pervious_area, impervious_area, total_area = calculate_ratio(NAIP_folder_path, tif_folder_path)

        writer.writerow([city, f'{pervious_ratio}', f'{impervious_ratio}', f'{pervious_area}', f'{impervious_area}', f'{total_area}'])

        print(city, 'Done!')


