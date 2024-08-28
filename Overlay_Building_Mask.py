import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import os
from shapely.geometry import box
import numpy as np

def overlay_buildings_on_tifs(buildings_geojson_path, tif_folder_path, output_folder):
    """
    将建筑物多边形覆盖到TIF图像上，覆盖区域的像素值设置为1。

    参数:
    - buildings_geojson_path: 建筑物GeoJSON文件的路径。
    - tif_folder_path: 包含TIF文件的文件夹路径。
    """
    # 读取建筑物GeoJSON文件
    buildings_gdf = gpd.read_file(buildings_geojson_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历TIF文件
    for tif_filename in os.listdir(tif_folder_path):
        if tif_filename.endswith('.tif'):
            tif_path = os.path.join(tif_folder_path, tif_filename)

            # 读取TIF文件
            with rasterio.open(tif_path) as tif:
                if buildings_gdf.crs.to_epsg() == 4326:
                    buildings_gdf = buildings_gdf.to_crs(tif.crs)

                # 获取当前TIF文件的边界
                bounds = tif.bounds
                tile_bounds = box(*bounds)

                # 筛选出与当前瓦片相交的建筑物
                clipped_buildings = buildings_gdf[buildings_gdf.geometry.intersects(tile_bounds)]

                out_tif_path = os.path.join(output_folder, "_".join(tif_filename.split('_')[1:]))

                # 读取TIF图像数据，并应用mask
                image_data = tif.read(1)  # 假设我们只处理单波段数据

                image_data = np.where(image_data != 2, 0, 1)

                if clipped_buildings.empty:
                    # 如果没有建筑物，保存重分类后的图像并跳过当前迭代
                    print(f"No buildings found, saving reclassified tile to {out_tif_path}")
                else:
                    # 如果有建筑物，创建覆盖的mask并应用
                    mask_image = geometry_mask(clipped_buildings.geometry, transform=tif.transform, invert=True,
                                               out_shape=(tif.height, tif.width))
                    image_data[mask_image] = 1  # 将覆盖的像素值设置为1

                # 保存修改后的TIF文件
                out_meta = tif.meta.copy()
                out_meta.update({
                    'dtype': 'uint8',
                    'count': 1
                })

                with rasterio.open(out_tif_path, 'w', **out_meta) as out_tif:
                    print(out_tif_path)
                    out_tif.write(image_data, 1)


cities = [
    # 'Atlanta',
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
    'Memphis',
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
    # 'Phoenix',
    'NewYorkCity'
]

for city in cities:

    buildings_geojson_path = f'/workspace/ericyi/Surface Albedo/data/{city}/Buildings/{city}_buildings.geojson'
    tif_folder_path = f'/workspace/ericyi/Surface Albedo/data/{city}/512_Impervious_UNet_mask_new'
    output_folder = f'/workspace/ericyi/Surface Albedo/data/{city}/512_Impervious_UNet_building_mask_new'
    overlay_buildings_on_tifs(buildings_geojson_path, tif_folder_path, output_folder)


