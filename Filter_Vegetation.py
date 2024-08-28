import rasterio
import numpy as np

def apply_mask_to_naip(naip_path, mask_path, output_path):
    # 打开NAIP数据
    with rasterio.open(naip_path) as naip_src:
        naip_data = naip_src.read()
        naip_meta = naip_src.meta

    # 打开掩膜数据
    with rasterio.open(mask_path) as mask_src:
        mask_data = mask_src.read()

    # 检查维度是否匹配
    if naip_data.shape != mask_data.shape:
        raise ValueError("The dimensions of the NAIP data and the mask do not match.")

    # 逐像素相乘
    masked_naip_data = naip_data * mask_data

    # 保存结果
    with rasterio.open(output_path, 'w', **naip_meta) as dst:
        dst.write(masked_naip_data)

    print(f"Masked NAIP data saved to '{output_path}'")

naip_file = r"F:\Ericyi\Surface Albedo\data\PA\NAIP\NAIP_PA_83_4Bands.tif"
single_file = r"F:\Ericyi\Surface Albedo\data\PA\NDVI\NAIP_PA_83_NDVI_4bands.tif"
output_file = r"F:\Ericyi\Surface Albedo\data\PA\NAIP\NAIP_PA_83_NoVegetation.tif"
apply_mask_to_naip(naip_file, single_file, output_file)