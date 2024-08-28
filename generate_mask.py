import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
import numpy as np

# 输入文件路径
naip_image_path = r'/data/Yi/Surface Albedo/data/LA/NAIP/NAIP_LA_Roof83.tif'
roof_shapefile_path = r'/data/Yi/Surface Albedo/data/LA/Albedo/Albedo_fixed_83.shp'
output_image_path = r'/data/Yi/Surface Albedo/data/LA/NAIP\NAIP_LA_Roof83_output.tif'

# 打开NAIP影像
with rasterio.open(naip_image_path) as src:
    naip_data = src.read(1)  # 读取影像数据
    naip_transform = src.transform  # 获取变换信息

# 打开roof shapefile
roof_data = gpd.read_file(roof_shapefile_path)

# 创建输出影像
output_data = np.zeros_like(naip_data, dtype=np.float32)

# 遍历每个roof并处理
for idx, roof in roof_data.iterrows():
    # 获取roof的CALIB_SR值
    calib_sr = roof['CALIB_SR']

    # 使用geometry_mask函数创建一个布尔掩码，用于标识与roof相交的像素
    mask = geometry_mask([roof.geometry], out_shape=output_data.shape, transform=naip_transform, invert=True)

    coordinates = np.column_stack(np.where(mask))

    # 使用掩码将CALIB_SR值赋值给output_data
    output_data[mask] = calib_sr
    print(idx)


# 将输出数据保存为TIFF
with rasterio.open(output_image_path, 'w', driver='GTiff', height=output_data.shape[0], width=output_data.shape[1],
                   count=1, dtype=np.float32, crs=src.crs, transform=naip_transform) as dst:
    dst.write(output_data, 1)
