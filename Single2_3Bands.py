import rasterio

# 打开单波段的TIF文件
with rasterio.open(r"F:\Ericyi\Surface Albedo\data\PA\NDVI\NAIP_PA_83_NDVI_mask.tif") as src:
    single_data = src.read(1)  # 读取第一个波段的数据
    single_meta = src.meta

# 更新元数据，使其成为三波段的TIF文件
single_meta['count'] = 4

# 保存新的三波段TIF文件
with rasterio.open(r"F:\Ericyi\Surface Albedo\data\PA\NDVI\NAIP_PA_83_NDVI_4bands.tif", 'w', **single_meta) as dest:
    for i in range(4):
        dest.write(single_data, i+1)
