import os
import pandas as pd
from sklearn.model_selection import train_test_split


def create_dataset_split(base_dir, cities):
    data = []
    for city in cities:
        naip_dir = os.path.join(base_dir, city, '1024_Pervious', 'naip_pervious')
        albedo_dir = os.path.join(base_dir, city, '1024_Pervious', 'pervious_albedo')

        naip_files = [os.path.join(naip_dir, f) for f in os.listdir(naip_dir)]
        albedo_files = [os.path.join(albedo_dir, f) for f in os.listdir(albedo_dir)]

        # Assume naip_files and albedo_files are matched by filename sorting
        data.extend(zip(naip_files, albedo_files, [city] * len(naip_files)))

    # Create DataFrame
    df = pd.DataFrame(data, columns=['naip_path', 'albedo_path', 'city'])

    # Split data into training and testing
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save to CSV
    train_df.to_csv(os.path.join(base_dir, 'train_files.csv'), index=False)
    test_df.to_csv(os.path.join(base_dir, 'test_files.csv'), index=False)

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

base_dir = '/workspace/ericyi/Surface Albedo/data'
create_dataset_split(base_dir, cities)