import shutil
import random
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import rasterio
from torch.utils.data.dataset import random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

seed = 42  # You can choose any number you want here
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


class ImageDataset(Dataset):
    def __init__(self, naip_dir, roof_dir):
        self.naip_files = [os.path.join(naip_dir, f) for f in os.listdir(naip_dir)]
        self.roof_files = [os.path.join(roof_dir, f) for f in os.listdir(roof_dir)]

    def __len__(self):
        return len(self.naip_files)

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
        # 'NewYorkCity'
    ]

for city in cities:
    naip_dir = fr'/workspace/ericyi/Surface Albedo/data/{city}/1024_Pervious/naip_pervious'
    roof_dir = fr'/workspace/ericyi/Surface Albedo/data/{city}/1024_Pervious/pervious_albedo'

    # Step 1: Create necessary directories
    train_naip_dir = f'/workspace/ericyi/Surface Albedo/data/{city}/1024_Pervious/naip_pervious_train'
    train_roof_dir = f'/workspace/ericyi/Surface Albedo/data/{city}/1024_Pervious/pervious_albedo_train'
    test_naip_dir = f'/workspace/ericyi/Surface Albedo/data/{city}/1024_Pervious/naip_pervious_test'
    test_roof_dir = f'/workspace/ericyi/Surface Albedo/data/{city}/1024_Pervious/pervious_albedo_test'

    os.makedirs(train_naip_dir, exist_ok=True)
    os.makedirs(train_roof_dir, exist_ok=True)
    os.makedirs(test_naip_dir, exist_ok=True)
    os.makedirs(test_roof_dir, exist_ok=True)


    # Step 2: Copy files to new directories based on indices from random_split
    dataset = ImageDataset(naip_dir, roof_dir)
    print(len(dataset))
    # Splitting the dataset 80:20
    train_len = int(0.8 * len(dataset))
    print(train_len)
    val_len = len(dataset) - train_len
    print(val_len)
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    # For training set
    for idx in train_dataset.indices:

        shutil.copy(dataset.naip_files[idx], os.path.join(train_naip_dir, os.path.basename(dataset.naip_files[idx])))
        shutil.copy(dataset.roof_files[idx], os.path.join(train_roof_dir, os.path.basename(dataset.roof_files[idx])))
        print(city, idx)
    # For validation set
    for idx in val_dataset.indices:
        shutil.copy(dataset.naip_files[idx], os.path.join(test_naip_dir, os.path.basename(dataset.naip_files[idx])))
        shutil.copy(dataset.roof_files[idx], os.path.join(test_roof_dir, os.path.basename(dataset.roof_files[idx])))