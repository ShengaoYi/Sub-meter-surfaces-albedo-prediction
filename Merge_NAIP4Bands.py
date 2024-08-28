import rasterio
from rasterio.merge import merge
import os

# 指定文件夹路径
dir_path = r"/workspace/ericyi/Surface Albedo/data/NewYorkCity/NAIP4"

# 获取文件夹中的所有TIFF文件
tiff_files = [os.path.join(dir_path, filename) for filename in os.listdir(dir_path) if filename.endswith('.tif')]

# 读取所有的TIFF文件
src_files_to_mosaic = [rasterio.open(fp) for fp in tiff_files]

# 使用merge方法进行合并
mosaic, out_trans = merge(src_files_to_mosaic)

# 输出文件路径
output_file = '/workspace/ericyi/Surface Albedo/data/NewYorkCity/NAIP_NYC_new.tif'

# 保存合并后的影像
with rasterio.open(output_file, 'w', driver='GTiff',
                   height=mosaic.shape[1], width=mosaic.shape[2],
                   count=src_files_to_mosaic[0].count,
                   dtype=mosaic.dtype,
                   crs=src_files_to_mosaic[0].crs,
                   transform=out_trans,
                    compress='lzw',
                    BIGTIFF='YES'
                   ) as dest:
    dest.write(mosaic)

# 关闭所有原始文件
for src in src_files_to_mosaic:
    src.close()

print("Images have been merged!")