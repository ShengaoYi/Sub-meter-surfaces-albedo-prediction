import geopandas as gpd
import os
from shapely.geometry import shape

root = r'/workspace/ericyi/Surface Albedo/data'
buildings_dir = os.path.join(root, 'Buildings')
cities = [
'SaltLakeCity',
'SanAntonio',
'SanDiego',
'SanFrancisco',
'Seattle',
'StLouis','Houston','NewYorkCity']
# 这里假设你已经有了一个城市到州的映射
city_to_state = {
    'Atlanta': 'Georgia',
    'Austin': 'Texas',
    'Baltimore': 'Maryland',
    'Boston': 'Massachusetts',
    'Charlotte': 'NorthCarolina',
    'Chicago': 'Illinois',
    'Cleveland': 'Ohio',
    'DC': 'DistrictofColumbia',
    'Dallas': 'Texas',
    'Denver': 'Colorado',
    'Detroit': 'Michigan',
    'Indianapolis': 'Indiana',
    'LasVegas': 'Nevada',
    'LosAngeles': 'California',
    'Louisville': 'Kentucky',
    'Memphis': 'Tennessee',
    'Miami': 'Florida',
    'Milwaukee': 'Wisconsin',
    'Minneapolis': 'Minnesota',
    'Nashville': 'Tennessee',
    'OklahomaCity': 'Oklahoma',
    'Philadelphia': 'Pennsylvania',
    'Phoenix': 'Arizona',
    'Pittsburgh': 'Pennsylvania',
    'Richmond': 'Virginia',
    'Sacramento': 'California',
    'SaltLakeCity': 'Utah',
    'SanAntonio': 'Texas',
    'SanDiego': 'California',
    'SanFrancisco': 'California',
    'Seattle': 'Washington',
    'StLouis': 'Missouri',
    'Houston': 'Texas',
    'NewYorkCity': 'NewYork'
}


for city in cities:
    print(f'Processing {city}')

    # 根据映射找到州的GeoJSON文件
    state_name = city_to_state[city]
    state_file = os.path.join(buildings_dir, state_name + '.geojson')

    # 读取州的建筑物文件
    state_buildings = gpd.read_file(state_file)

    # 读取城市的边界文件
    city_dir = os.path.join(root, city, 'Boundary')
    city_buildings_dir = os.path.join(root, city, 'Buildings')
    if not os.path.exists(city_buildings_dir):
        os.makedirs(city_buildings_dir)

    boundary_file = os.path.join(city_dir, city + '_boundary.shp')
    city_boundary = gpd.read_file(boundary_file)

    # 使用城市边界剪裁州建筑物数据
    clipped_buildings = gpd.clip(state_buildings, city_boundary)

    # 将剪裁后的建筑物数据保存为Shapefile，文件名为'{city}_buildings.shp'
    output_path = os.path.join(city_buildings_dir, f'{city}_buildings.geojson')
    clipped_buildings.to_file(output_path, driver='GeoJSON')

