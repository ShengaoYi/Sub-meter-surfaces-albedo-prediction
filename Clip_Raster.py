import geopandas as gpd
import rasterio
from rasterio.mask import mask

# 读取 Shapefile
city = 'NewYorkCity'
shp_file = fr'/workspace/ericyi/Surface Albedo/data/{city}/Boundary/{city}_boundary.shp'
shapefile = gpd.read_file(shp_file)

# 读取 TIFF 文件
tif_file = fr'/workspace/ericyi/Surface Albedo/data/{city}/NAIP4/NAIP_{city}_2021_26916.tif'


with rasterio.open(tif_file) as src:
    raster_crs = src.crs
    shapefile = shapefile.to_crs(raster_crs)
    # Convert the geometries from the GeoDataFrame to GeoJSON format
    shapes = [feature["geometry"] for feature in shapefile.to_dict(orient="records")]

    # Clip the raster with the shapes
    out_image, out_transform = mask(src, shapes, crop=True)
    out_meta = src.meta

# 设置剪裁后的文件的元数据（更新维度、转换和坐标系统）
out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform,
                 'compress': 'lzw',  # Add LZW compression here
                 'BIGTIFF': 'YES'
                 })

# 写入剪裁后的 raster 文件
with rasterio.open(fr'/workspace/ericyi/Surface Albedo/data/{city}/NAIP/NAIP_{city}_2021_26916.tif', 'w', **out_meta) as dest:
    dest.write(out_image)
