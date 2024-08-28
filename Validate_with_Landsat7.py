import rasterio
import numpy as np
from rasterio.windows import from_bounds
from rasterio.windows import Window


def calculate_metrics(file_30m, file_high_res):
    with rasterio.open(file_30m) as src_30m, rasterio.open(file_high_res) as src_high_res:
        width = src_30m.width
        height = src_30m.height

        truth_values = []
        predicted_values = []

        for i in range(width):
            for j in range(height):
                window_30m = Window(i, j, 1, 1)
                albedo_30m = src_30m.read(1, window=window_30m)[0, 0]

                if albedo_30m <= 0.05:
                    continue

                # 获取30米像素的边界
                bounds = src_30m.window_bounds(window_30m)

                # 根据边界获取对应的高分辨率像素区域
                window_high_res = from_bounds(*bounds, src_high_res.transform)
                albedo_high_res = src_high_res.read(1, window=window_high_res)

                # 计算高分辨率像素区域的平均值
                mean_albedo_high_res = np.mean(albedo_high_res)

                if np.isnan(mean_albedo_high_res) or np.isnan(albedo_30m):
                    continue

                truth_values.append(albedo_30m)
                predicted_values.append(mean_albedo_high_res)

                print(i, j, albedo_30m, mean_albedo_high_res)

        truth_values = np.array(truth_values)
        predicted_values = np.array(predicted_values)

        N = len(truth_values)
        mean_truth = np.mean(truth_values)
        mean_predicted = np.mean(predicted_values)

        bias = np.mean(predicted_values - truth_values)
        rmse = np.sqrt(np.mean((predicted_values - truth_values) ** 2))
        rrmse = (rmse / mean_truth) * 100

        numerator = np.sum((predicted_values - mean_predicted) * (truth_values - mean_truth))
        denominator = np.sqrt(
            np.sum((predicted_values - mean_predicted) ** 2) * np.sum((truth_values - mean_truth) ** 2))
        r_squared = (numerator / denominator) ** 2

        correlation_matrix = np.corrcoef(predicted_values, truth_values)
        correlation_coefficient = correlation_matrix[0, 1]



        return bias, rmse, rrmse, r_squared, correlation_coefficient

city = 'Atlanta'

# 示例使用
file_30m = f'/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Albedo_Landsat7/Landsat7_Atlanta_2022_26916_Albedo.tif'
file_1m = f'/workspace/ericyi/Surface Albedo/data/{city}/Albedo/{city}_Albedo_UNet.tif'
bias, rmse, rrmse, r_squared, correlation_coefficient  = calculate_metrics(file_30m, file_1m)

print(f'Bias: {bias}')
print(f'RMSE: {rmse}')
print(f'rRMSE: {rrmse}%')
print(f'R²: {r_squared}')
print(f'Correlation Coefficient: {correlation_coefficient}')