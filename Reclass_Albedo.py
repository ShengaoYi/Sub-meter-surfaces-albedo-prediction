import rasterio

def correct_negative_pixels(input_tif, output_tif):
    with rasterio.open(input_tif, 'r') as src:
        # 读取数据
        band1 = src.read(1)

        # 找出小于0的值，并将其设置为0
        band1[band1 < 0.03] = 0

        # 使用原始的metadata来保存新的tiff
        meta = src.meta

        with rasterio.open(output_tif, 'w', **meta) as dst:
            dst.write(band1, 1)

# 使用函数
input_tif_path = r'F:\Ericyi\Surface Albedo\data\PA\Albedo\PA_Albedo_UNet_NoVegetation.tif'
output_tif_path = r'F:\Ericyi\Surface Albedo\data\PA\Albedo\PA_Albedo_UNet_NoVegetation_Reclass.tif'
correct_negative_pixels(input_tif_path, output_tif_path)
