# -*- coding: utf-8 -*-
'''
@Time    : 2023/6/15 10:58
@Author  : Ericyi
@File    : Data_Process.py

'''
import numpy as np
import rasterio
from shapely import Point
import geopandas as gpd

lst_data = rasterio.open(r'E:\Paper\ICLR_Urban_Heat\data\LST\LST_LA_3310.tif')
census_tracts = gpd.read_file(r'E:\Paper\ICLR_Urban_Heat\data\ACS\2022_ACS_LA_city.geojson')

def calculate_essential_workers(census_tracts):

    census_tracts['Unemployed_p'] = census_tracts['Unemployed'] / (census_tracts['Employed'] + census_tracts['Unemployed']) * 100
    census_tracts['Bachelors'] = census_tracts['MaleBachelors'] + census_tracts['FemaleBachelors']

    census_tracts['SNAP'] = census_tracts['SNAP'] / census_tracts['TotalHHd'] * 100

    census_tracts["PopDensity"] = census_tracts["TotalPop"] / census_tracts['Area_km2']

    # 计算每个族群的比例
    census_tracts['White_prop'] = census_tracts['Whites'] / census_tracts['TotalPop']
    census_tracts['Black_prop'] = census_tracts['Black'] / census_tracts['TotalPop']
    census_tracts['Native_American_prop'] = census_tracts['Native_American'] / census_tracts['TotalPop']
    census_tracts['Asian_prop'] = census_tracts['Asian'] / census_tracts['TotalPop']
    census_tracts['OtherRaces_prop'] = census_tracts['OtherRaces'] / census_tracts['TotalPop']

    # 计算香农多样性指数
    census_tracts['Shannon_diversity'] = -(census_tracts['White_prop'] * np.log(census_tracts['White_prop']) +
                                 census_tracts['Black_prop'] * np.log(census_tracts['Black_prop']) +
                                 census_tracts['Native_American_prop'] * np.log(census_tracts['Native_American_prop']) +
                                 census_tracts['Asian_prop'] * np.log(census_tracts['Asian_prop']) +
                                 census_tracts['OtherRaces_prop'] * np.log(census_tracts['OtherRaces_prop'])).replace(-np.inf, 0)

    # 计算辛普森多样性指数
    census_tracts['Simpson_diversity'] = 1 - (census_tracts['White_prop'] ** 2 +
                                    census_tracts['Black_prop'] ** 2 +
                                    census_tracts['Native_American_prop'] ** 2 +
                                    census_tracts['Asian_prop'] ** 2 +
                                    census_tracts['OtherRaces_prop'] ** 2)

    columns_to_remove = [
            "Employed", "Unemployed", "MaleBachelors", "FemaleBachelors", 'White_prop', 'Black_prop',
            'Native_American_prop', 'Asian_prop', 'OtherRaces_prop'
        ]

    census_tracts = census_tracts.drop(columns=columns_to_remove)

    # 计算平均住房负担
    census_tracts['Housing_Burden'] = census_tracts['MedRent'] / census_tracts['MedHHInc']

    # 计算每个家庭的平均人数
    census_tracts['Household_Size'] = census_tracts['TotalPop'] / census_tracts['TotalHHd']

    columns_to_update = ["Whites", "Black", "Native_American", "Asian", "OtherRaces", "povBelow100", "pov100_150",
                         "WFH", "Bachelors", "TotalWorkPop"]

    for column in columns_to_update:
        census_tracts[column] = census_tracts[column] / census_tracts["TotalPop"] * 100

    return census_tracts

def calculate_crime_density(census_tracts, crime):
    # Perform spatial joins for each crime type
    felony_crime = crime[crime['Type'] == 'Felony']
    misdemeanor_crime = crime[crime['Type'] == 'Misdemeanor']
    violation_crime = crime[crime['Type'] == 'Violation']

    tracts_with_felony = gpd.sjoin(census_tracts, felony_crime, how='left', predicate='intersects')
    tracts_with_misdemeanor = gpd.sjoin(census_tracts, misdemeanor_crime, how='left', predicate='intersects')
    tracts_with_violation = gpd.sjoin(census_tracts, violation_crime, how='left', predicate='intersects')

    # Count crime points per tract
    felony_count = tracts_with_felony.groupby('GEOID').size().reset_index(name='Felony_Count')
    misdemeanor_count = tracts_with_misdemeanor.groupby('GEOID').size().reset_index(name='Misdemeanor_Count')
    violation_count = tracts_with_violation.groupby('GEOID').size().reset_index(name='Violation_Count')

    # Merge with census tracts
    census_tracts = census_tracts.merge(felony_count, on='GEOID', how='left')
    census_tracts = census_tracts.merge(misdemeanor_count, on='GEOID', how='left')
    census_tracts = census_tracts.merge(violation_count, on='GEOID', how='left')

    census_tracts['Felony_density_per_km2'] = census_tracts['Felony_Count'] / census_tracts['Area_km2']
    census_tracts['Misdemeanor_density_per_km2'] = census_tracts['Misdemeanor_Count'] / census_tracts['Area_km2']
    census_tracts['Violation_density_per_km2'] = census_tracts['Violation_Count'] / census_tracts['Area_km2']

    columns_to_remove = [
        "Felony_Count", "Misdemeanor_Count", "Violation_Count"
    ]

    census_tracts = census_tracts.drop(columns=columns_to_remove)

    return census_tracts

# Set the CRS for both GeoDataFrames to EPSG:3310 for accurate area calculations
crs_epsg = 'EPSG:3310'
# lst_data = lst_data.to_crs(crs_epsg)
census_tracts = census_tracts.to_crs(crs_epsg)

census_tracts['Area_km2'] = census_tracts['geometry'].area / 1e6

census_tracts = calculate_essential_workers(census_tracts)

SVI_points_file = 'E:\Paper\ICLR_Urban_Heat\data\SVI\LA_seg_results.shp'
indices_columns = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic li', 'traffic si', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle']

def aggregate_SVI(tracts, points):
    """计算街景语义分割结果的各个指标"""
    points['SVI'] = points['sky']
    points['GVI'] = points['vegetation']
    points['Enclosure'] = points['building'] + points['wall'] + points['vegetation'] - points['terrain']
    points['Walkability'] = points['sidewalk'] + points['terrain'] + points['person']
    points['Street_spaciousness'] = points['road']

    tracts_with_points = gpd.sjoin(tracts, points, how='left', predicate='intersects')

    # 计算每个tract的指标平均值
    aggregate_functions = {
        'SVI': 'mean',
        'GVI': 'mean',
        'Enclosure': 'mean',
        'Walkability': 'mean',
        'Street_spaciousness': 'mean'
    }
    tracts_with_points = tracts_with_points.reset_index()
    tracts_avg = tracts_with_points.groupby('GEOID').agg(aggregate_functions)

    # 把计算出的平均值添加到原始tracts数据框中
    # 这里的on参数应该是tracts数据框的索引名称，如果它有一个特定的名字，应该替换'index'
    tracts = tracts.merge(tracts_avg, left_on='GEOID', right_index=True, how='left')

    return tracts

SVI_points = gpd.read_file(SVI_points_file)
SVI_points = SVI_points.to_crs(crs_epsg)
census_tracts = aggregate_SVI(census_tracts, SVI_points)

census_tracts = census_tracts.to_crs('EPSG:4326')

census_tracts.to_file(r'E:\Paper\ICLR_Urban_Heat\data\Total_result\2022_ACS_with_LST.geojson')
