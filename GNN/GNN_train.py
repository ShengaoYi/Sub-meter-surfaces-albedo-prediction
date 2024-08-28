import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import rasterio
from torch.utils.data.dataset import random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from GNN import GNNModel
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score

train_dir_naip = r'/data/Yi/Surface Albedo/data/LA/512/naip_train'
train_dir_roof = r'/data/Yi/Surface Albedo/data/LA/512/roof_output'

val_dir_naip = r'/data/Yi/Surface Albedo/data/LA/512/naip_test'
val_dir_roof = r'/data/Yi/Surface Albedo/data/LA/512/roof_output'


class ImageDataset(Dataset):
    def __init__(self, naip_dir, roof_dir):
        self.naip_files = [os.path.join(naip_dir, f) for f in os.listdir(naip_dir)]
        self.roof_files = [os.path.join(roof_dir, f) for f in os.listdir(roof_dir)]

    def __len__(self):
        return len(self.naip_files)

    def __getitem__(self, idx):
        with rasterio.open(self.naip_files[idx]) as src:
            naip_image = src.read().transpose((1, 2, 0)) # Convert from CxHxW to HxWxC

        with rasterio.open(self.roof_files[idx]) as src:
            roof_image = src.read()[0]  # Assuming roof_image is single channeled
        # if naip_image.shape != (2048, 2048, 3) or (roof_image.shape != (2048, 2048) and roof_image.ndim == 2):
        #     print(f"Inconsistent shape detected!")
        #     print(f"NAIP image: {self.naip_files[idx]}, Shape: {naip_image.shape}")
        #     print(f"Roof image: {self.roof_files[idx]}, Shape: {roof_image.shape}")
        #     raise ValueError("Detected image with inconsistent shape!")

        return torch.tensor(naip_image.transpose((2, 0, 1)), dtype=torch.float32), torch.tensor(roof_image[np.newaxis, ...], dtype=torch.float32)

# New weighted loss function
def weighted_mse_loss(input, target, weight_for_zeros=0.1, weight_for_nonzeros=1.0):
    weights = torch.ones_like(target) * weight_for_nonzeros
    weights[target == 0] = weight_for_zeros
    return ((input - target) ** 2 * weights).mean()

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


train_dataset = ImageDataset(train_dir_naip, train_dir_roof)
val_dataset = ImageDataset(val_dir_naip, val_dir_roof)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)  # No need to shuffle validation set

# Initialize the TensorBoard writer
writer = SummaryWriter()

model = GNNModel(input_channels=3).to(device)
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model, device_ids=[3, 4])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

num_epochs = 1000

def compute_metrics(pred, gt):
    # 计算指标，确保在计算之前过滤掉ground truth为0的像素
    # mask = gt != 0
    # pred, gt = pred[mask], gt[mask]
    mse = mean_squared_error(gt, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(gt, pred)
    # mape = np.mean(np.abs((gt - pred) / gt)) * 100
    r2 = r2_score(gt, pred)

    return mse, rmse, mae, r2

# Training loop
for epoch in range(num_epochs):
    model.train()  # set the model to training mode
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
        writer.add_scalar('training_loss', loss.item(), epoch * len(train_dataloader) + batch_idx)

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

    print(f"Average Validation MSE: {avg_mse:.4f}, RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}, R^2: {avg_r2:.4f}")

    print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.7f}')

    # Step the scheduler
    # scheduler.step(avg_val_loss)

    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), f'/data/Yi/Surface Albedo/codes/model/gnn_model_epoch_{epoch + 1}_512.pth')
# Save
torch.save(model.state_dict(), '/data/Yi/Surface Albedo/codes/model/gnn_model_512.pth')

# Load
# model = UNet()
# model.load_state_dict(torch.load('/data/Yi/Surface Albedo/codes/model/loss_unet_model.pth'))
# model.eval()

# Evaluate on test data if needed
# total_loss = 0
# model.eval()  # set the model to evaluation mode
# with torch.no_grad():
#     for batch_idx, (naip_images, roof_images) in enumerate(test_dataloader):
#         naip_images, roof_images = naip_images.to(device), roof_images.to(device)
#
#         outputs = model(naip_images)
#         # loss = criterion(outputs, roof_images)
#         loss = weighted_mse_loss(outputs, roof_images)
#         total_loss += loss.item()
#
#         # Log the test loss to TensorBoard
#         writer.add_scalar('test_loss', loss.item(), batch_idx)
#
# avg_test_loss = total_loss / len(test_dataloader)
# print(f'Average test loss: {avg_test_loss:.4f}')

# Close the TensorBoard writer
writer.close()
