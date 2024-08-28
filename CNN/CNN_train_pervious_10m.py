import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import rasterio
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
from CNN import SimpleCNN
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import random
seed = 42  # You can choose any number you want here
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
#
class ImageDataset(Dataset):
    def __init__(self, naip_files, roof_files):
        self.naip_files = naip_files
        self.roof_files = roof_files


    def __len__(self):
        return len(self.naip_files)

    def __getitem__(self, idx):
        with rasterio.open(self.naip_files[idx]) as src:
            naip_image = src.read().astype('float32') # Convert from CxHxW to HxWxC

        with rasterio.open(self.roof_files[idx]) as src:
            roof_image = src.read().astype('float32')[0]  # Assuming roof_image is single channeled
            roof_image = np.nan_to_num(roof_image, nan=0.0)


        mask = np.all(naip_image == 0, axis=0).astype(np.float32)

        mask = 1 - mask

        masked_roof_image = roof_image * mask
        # print(naip_image, masked_roof_image)

        naip_image = naip_image.transpose((1, 2, 0))  # Convert from CxHxW to HxWxC
        naip_image[(naip_image == 256).all(axis=2)] = 0

        return torch.tensor(naip_image.transpose((2, 0, 1)), dtype=torch.float32), torch.tensor(masked_roof_image[np.newaxis, ...], dtype=torch.float32)

    def get_filename(self, idx):
        """Returns the filename for a given index."""
        return self.naip_files[idx], self.roof_files[idx]

def list_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)]

naip_files = list_files("/workspace/ericyi/Surface Albedo/data/Philadelphia/10m/1024_Pervious/naip_pervious")  # Update with your path
roof_files = list_files("/workspace/ericyi/Surface Albedo/data/Philadelphia/10m/1024_Pervious/pervious_albedo")  # Update with your path
# 确保文件列表有相同的顺序，这一步很重要
naip_files.sort()
roof_files.sort()

# 分割数据，比如用80%数据作为训练集，20%作为验证集
split_index = int(0.8 * len(naip_files))

train_naip_files = naip_files[:split_index]
train_roof_files = roof_files[:split_index]
val_naip_files = naip_files[split_index:]
val_roof_files = roof_files[split_index:]

### 步骤 3: 创建数据集和数据加载器
train_dataset = ImageDataset(train_naip_files, train_roof_files)
val_dataset = ImageDataset(val_naip_files, val_roof_files)

print(len(train_dataset), len(val_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')

log_dir = f'runs/training_CNN_pervious_1024_{current_time}'
# Initialize the TensorBoard writer
writer = SummaryWriter(log_dir)

model = SimpleCNN(input_channels=4, output_channels=1).to(device)

# if torch.cuda.device_count() > 1:
#     print("Using", torch.cuda.device_count(), "GPUs!")
#     model = nn.DataParallel(model, device_ids=[0, 1])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

num_epochs = 150

def compute_metrics(pred, gt):
    # 计算指标，确保在计算之前过滤掉ground truth为0的像素
    # mask = gt != 0
    # pred, gt = pred[mask], gt[mask]
    mse = mean_squared_error(gt, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(gt, pred)
    r2 = r2_score(gt, pred)

    return mse, rmse, mae, r2

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (naip_images, roof_images) in enumerate(train_dataloader):
        naip_images, roof_images = naip_images.to(device), roof_images.to(device)

        optimizer.zero_grad()

        outputs = model(naip_images)

        loss = criterion(outputs, roof_images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    # Log the training loss to TensorBoard
    writer.add_scalar('training_loss', total_loss, epoch)

    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.7f}')

    # Validation
    model.eval()
    total_val_loss = 0

    mse_vals, rmse_vals, mae_vals, r2_vals = [], [], [], []

    with torch.no_grad():
        for naip_images, roof_images in val_dataloader:
            naip_images, roof_images = naip_images.to(device), roof_images.to(device)

            outputs = model(naip_images)
            loss = criterion(outputs, roof_images)

            total_val_loss += loss.item()

            for output, target in zip(outputs.cpu().numpy(), roof_images.cpu().numpy()):
                mse, rmse, mae, r2 = compute_metrics(output.flatten(), target.flatten())
                mse_vals.append(mse)
                rmse_vals.append(rmse)
                mae_vals.append(mae)
                r2_vals.append(r2)

    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_mse = np.mean(mse_vals)
    avg_rmse = np.mean(rmse_vals)
    avg_mae = np.mean(mae_vals)
    avg_r2 = np.mean(r2_vals)

    writer.add_scalar('validation_loss', avg_val_loss, epoch)
    writer.add_scalar('validation_mse', avg_mse, epoch)
    writer.add_scalar('validation_rmse', avg_rmse, epoch)
    writer.add_scalar('validation_mae', avg_mae, epoch)
    writer.add_scalar('validation_r2', avg_r2, epoch)

    print(f"Validation Metrics - MSE: {avg_mse:.4f}, RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}, R^2: {avg_r2:.4f}")
    print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.7f}')


    # Step the scheduler
    # scheduler.step(avg_val_loss)

    if (epoch + 1) % 20 == 0:
        torch.save(model.state_dict(), rf'/workspace/ericyi/Surface Albedo/models/CNN/UNet_epoch_{epoch + 1}_pervious_1024.pth')
