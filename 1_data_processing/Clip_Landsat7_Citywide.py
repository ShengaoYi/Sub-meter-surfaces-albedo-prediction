import geopandas as gpd
import rasterio
import os
from rasterio.mask import mask
cities = [
    # 'Atlanta',
    # 'Austin',
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
# 'NewYorkCity'
    'Phoenix',
    ]

root = r'/workspace/ericyi/Surface Albedo/data'

for city in cities:
    print(city)
    city_dir = os.path.join(root, city)

    # 读取 Shapefile
    boundary_dir = os.path.join(city_dir, 'Boundary')
    boundary_file = os.path.join(boundary_dir, city + '_boundary.shp')
    shapefile = gpd.read_file(boundary_file)

    NAIP_dir = os.path.join(city_dir, 'Landsat7')


    files = os.listdir(NAIP_dir)

    if len(files) == 1:
        tile_name = files[0]

        tif_file = os.path.join(NAIP_dir, tile_name)

        with rasterio.open(tif_file) as src:
            raster_crs = src.crs  # 读取栅格数据的坐标系统
            epsg_code = raster_crs.to_epsg()  # 获取EPSG代码

            # 如果 Shapefile 和 Raster 的坐标系统不匹配，则重投影 Shapefile
            if shapefile.crs != raster_crs:
                shapefile = shapefile.to_crs(raster_crs)

            # Convert the geometries from the GeoDataFrame to GeoJSON format
            shapes = [feature["geometry"] for feature in shapefile.to_dict(orient="records")]

            # Clip the raster with the shapes
            out_image, out_transform = mask(src, shapes, crop=True)
            out_meta = src.meta

        new_tile = os.path.join(city_dir, tile_name)
        # 设置剪裁后的文件的元数据（更新维度、转换和坐标系统）
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # 写入剪裁后的 raster 文件
        with rasterio.open(new_tile, 'w', **out_meta) as dest:
            dest.write(out_image)
