import os
import numpy as np
import rasterio
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 路径
pred_folder = r"/workspace/ericyi/Surface Albedo/data/SJ/512/naip_test_Albedo_UNet_512_output_50"
gt_folder = r"/workspace/ericyi/Surface Albedo/data/SJ/512/roof_test"


def compute_metrics(pred, gt):
    # 计算指标，确保在计算之前过滤掉ground truth为0的像素
    # mask = gt != 0
    # pred, gt = pred[mask], gt[mask]
    mse = mean_squared_error(gt, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(gt, pred)

    r2 = r2_score(gt, pred)

    return mse, rmse, mae, r2


# 对每张图片计算指标
all_mse, all_rmse, all_mae, all_mape, all_r2 = [], [], [], [], []
for filename in os.listdir(pred_folder):
    if filename.startswith("pred_") and filename.endswith(".tif"):

        gt_filename = filename[len("pred_"):]
        pred_path = os.path.join(pred_folder, filename)
        gt_path = os.path.join(gt_folder, gt_filename)

        if os.path.exists(gt_path):  # 确保存在对应的真值文件
            with rasterio.open(pred_path) as src:
                pred_img = src.read(1)  # Assuming single channel

            with rasterio.open(gt_path) as src:
                gt_img = src.read(1)  # Assuming single channel

            mse, rmse, mae, r2 = compute_metrics(pred_img.flatten(), gt_img.flatten())
            all_mse.append(mse)
            all_rmse.append(rmse)
            all_mae.append(mae)
            all_r2.append(r2)
            break

avg_mse = np.mean(all_mse)
avg_rmse = np.mean(all_rmse)
avg_mae = np.mean(all_mae)
avg_r2 = np.mean(all_r2)

print(f"Average MSE: {avg_mse:.5f}")
print(f"Average RMSE: {avg_rmse:.5f}")
print(f"Average MAE: {avg_mae:.5f}")
print(f"Average R^2: {avg_r2:.5f}")
