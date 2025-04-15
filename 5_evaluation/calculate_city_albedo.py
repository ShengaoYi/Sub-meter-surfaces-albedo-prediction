import os.path

import rasterio
import pandas as pd
import numpy as np


def calculate_nonzero_mean_albedo(filepath):
    """ 读取反照率数据并计算非零像素的平均值 """
    with rasterio.open(filepath) as src:
        data = src.read(1)  # 读取第一波段
        non_zero_pixels = data[data > 0]  # 选择非零像素
        if non_zero_pixels.size > 0:
            return np.mean(non_zero_pixels)
        else:
            return np.nan  # 如果没有非零像素，返回 NaN

# 示例用法
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
        'NewYorkCity'
    ]

csv_path = r'/workspace/ericyi/Surface Albedo/data/cities_impervious.csv'
df = pd.read_csv(csv_path)
df['Albedo'] = np.nan  # 初始化新列

for city in cities:
    file = fr"/workspace/ericyi/Surface Albedo/data/{city}/Albedo/{city}_Albedo_UNet.tif"
    mean_albedo = calculate_nonzero_mean_albedo(file)

    df.loc[df['City'] == city, 'Albedo'] = mean_albedo

    print(city, mean_albedo)

df.to_csv(csv_path, index=False)  # 将更新后的 DataFrame 写回 CSV
