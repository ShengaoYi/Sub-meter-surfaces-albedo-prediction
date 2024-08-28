import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import rasterio
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
from FCN import FCN
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
seed = 42  # You can choose any number you want here
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class ImageDataset(Dataset):
    def __init__(self, base_dir, cities, data_type):
        self.files = []
        for city in cities:
            naip_dir = os.path.join(base_dir, city, '512', f'naip_{data_type}')
            roof_dir = os.path.join(base_dir, city, '512', f'roof_{data_type}')
            naip_files = [os.path.join(naip_dir, f) for f in os.listdir(naip_dir)]
            roof_files = [os.path.join(roof_dir, f) for f in os.listdir(roof_dir)]
            self.files.extend(zip(naip_files, roof_files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        naip_file, roof_file = self.files[idx]

        with rasterio.open(naip_file) as src:
            naip_image = src.read(out_dtype='float32')

        with rasterio.open(roof_file) as src:
            roof_image = src.read(1, out_dtype='float32')  # Assuming roof_image is single channeled

        naip_image = naip_image.transpose((1, 2, 0))  # Convert from CxHxW to HxWxC
        naip_image[(naip_image == 256).all(axis=2)] = 0

        return torch.tensor(naip_image.transpose((2, 0, 1)), dtype=torch.float32), \
               torch.tensor(roof_image[np.newaxis, ...], dtype=torch.float32)

cities = ['LA', 'BA', 'SF', 'LB', 'SJ']

train_dataset = ImageDataset('/workspace/ericyi/Surface Albedo/data', cities, 'train')
val_dataset = ImageDataset('/workspace/ericyi/Surface Albedo/data', cities, 'test')

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')

log_dir = f'runs/training_FCN_all_cities_{current_time}'
# Initialize the TensorBoard writer
writer = SummaryWriter(log_dir)

model = FCN(input_channels=4, output_channels=1).to(device)

# if torch.cuda.device_count() > 1:
#     print("Using", torch.cuda.device_count(), "GPUs!")
#     model = nn.DataParallel(model, device_ids=[2, 3])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

num_epochs = 200

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
        # print(roof_images, outputs)
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

    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), rf'/workspace/ericyi/Surface Albedo/models/FCN/FCN_epoch_{epoch + 1}_all_cities_512.pth')
