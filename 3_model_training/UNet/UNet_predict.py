import os
import numpy as np
import torch
import torch.nn as nn
import rasterio
from Unet import UNet

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    model = UNet(input_channels=4).to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=[2, 3])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# def predict_image(model, input_path, output_path):
#     with rasterio.open(input_path) as src:
#         naip_image = src.read().astype('float32').transpose((1, 2, 0))
#         transform = src.transform  # 获取转换信息
#         crs = src.crs  # 获取坐标参考系统
#
#     naip_image[(naip_image == 256).all(axis=2)] = 0
#
#     naip_tensor = torch.tensor(naip_image.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0)
#
#     with torch.no_grad():
#         output = model(naip_tensor)
#
#         output_image = output.cpu().squeeze(0).squeeze(0).numpy()
#
#     with rasterio.open(output_path, 'w', driver='GTiff', height=output_image.shape[0], width=output_image.shape[1],
#                        count=1, dtype=output_image.dtype, crs=crs, transform=transform) as dst:
#         dst.write(output_image, 1)

def predict_image(model, input_path, output_path, mask_folder):
    # 构造掩模文件的路径
    mask_filename = "impervious_" + os.path.basename(input_path)
    mask_path = os.path.join(mask_folder, mask_filename)

    # 读取NAIP图像
    with rasterio.open(input_path) as src:
        naip_image = src.read().astype('float32')
        transform = src.transform
        crs = src.crs

    # 读取对应的掩模
    with rasterio.open(mask_path) as mask_src:
        mask = mask_src.read(1)

    # 将掩模扩展到NAIP图像的每个波段，并应用掩模
    mask_extended = np.repeat(mask[np.newaxis, :, :], naip_image.shape[0], axis=0)
    naip_image *= mask_extended

    # 处理NAIP图像，准备模型输入
    naip_tensor = torch.tensor(naip_image, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(naip_tensor)

        output_image = output.cpu().squeeze(0).squeeze(0).numpy()

    # 再次应用掩模，以确保仅impervious区域的预测值被保留
    output_image *= mask

    output_image[output_image < 0] = 0

    # 保存预测结果
    with rasterio.open(output_path, 'w', driver='GTiff', height=output_image.shape[0], width=output_image.shape[1],
                       count=1, dtype=output_image.dtype, crs=crs, transform=transform) as dst:
        dst.write(output_image, 1)

if __name__ == "__main__":
    MODEL_PATH = "/workspace/ericyi/Surface Albedo/models/UNet/UNet_epoch_100_all_cities_512.pth"

    cities = [
        # 'Austin',
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
        'NewYorkCity']

    for city in cities:

        INPUT_NAIP_FOLDER = fr"/workspace/ericyi/Surface Albedo/data/{city}/512"
        OUTPUT_FOLDER = fr"/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Albedo_UNet"
        MASK_FOLDER = fr"/workspace/ericyi/Surface Albedo/data/{city}/512_Impervious_UNet_building_mask"

        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

        model = load_model(MODEL_PATH)

        for filename in os.listdir(INPUT_NAIP_FOLDER):
            if filename.endswith(".tif"):
                input_path = os.path.join(INPUT_NAIP_FOLDER, filename)
                output_path = os.path.join(OUTPUT_FOLDER, "pred_" + filename)
                predict_image(model, input_path, output_path, MASK_FOLDER)
                print(f"Prediction for {filename} saved to {output_path}")

