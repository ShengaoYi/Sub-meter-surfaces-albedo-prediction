import os
import numpy as np
import torch
import torch.nn as nn
import rasterio
from Unet import UNet
from rasterio.windows import Window

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
def load_model(model_path):
    model = UNet(input_channels=4).to(device)
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model, device_ids=[2, 3])
    model.load_state_dict(torch.load(model_path))
    # model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def crop_image(image_data, crop_margin):
    """
    裁剪图像数据。
    :param image_data: 输入的图像数组（假设是多波段的numpy数组）。
    :param crop_margin: 从每边裁剪的像素数。
    :return: 裁剪后的图像数组。
    """
    # 计算窗口
    window = Window(crop_margin, crop_margin, image_data.shape[1] - 2 * crop_margin,
                    image_data.shape[0] - 2 * crop_margin)

    # 使用窗口裁剪图像
    cropped_image = image_data[window.row_off:window.row_off + window.height,
                    window.col_off:window.col_off + window.width]
    return cropped_image


crop_margin = 256

def predict_image(model, input_path, output_path):

    # 读取NAIP图像
    with rasterio.open(input_path) as src:
        naip_image = src.read().astype('float32')
        transform = src.transform
        crs = src.crs

    # 读取对应的掩模
    mask = np.all(naip_image == 0, axis=0).astype(np.float32)

    mask = 1 - mask

    # 处理NAIP图像，准备模型输入
    naip_tensor = torch.tensor(naip_image, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(naip_tensor)

        output_image = output.cpu().squeeze(0).squeeze(0).numpy()

    # 应用掩模，以确保仅pervious区域的预测值被保留
    output_image *= mask

    output_image[output_image < 0] = 0

    # 保存预测结果
    with rasterio.open(output_path, 'w', driver='GTiff', height=output_image.shape[0], width=output_image.shape[1],
                       count=1, dtype=output_image.dtype, crs=crs, transform=transform) as dst:
        dst.write(output_image, 1)

if __name__ == "__main__":
    MODEL_PATH = "/workspace/ericyi/Surface Albedo/models/UNet/UNet_epoch_20_pervious_1024.pth"

    cities = [
        # 'Austin',
        # 'Atlanta',
        # 'Baltimore',
        # 'Boston',
        # 'Charlotte',
        # 'Chicago',
        # 'Cleveland',
        # 'DC',
        # 'Dallas',
        # 'Denver',
        # 'Detroit',
        # 'Indianapolis',
        # 'LasVegas',
        # 'Louisville',
        # 'Memphis',
        # 'Miami',
        # 'Milwaukee',
        # 'Nashville',
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
        'NewYorkCity'
    ]

    for city in cities:

        INPUT_NAIP_FOLDER = fr"/workspace/ericyi/Surface Albedo/data/{city}/1024_Pervious/naip_pervious_new"
        OUTPUT_FOLDER = fr"/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Pervious_Albedo_UNet_Masked_1024_new"

        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

        model = load_model(MODEL_PATH)

        n = 0

        for filename in os.listdir(INPUT_NAIP_FOLDER):
            if filename.endswith(".tif"):
                n += 1
                torch.cuda.empty_cache()
                input_path = os.path.join(INPUT_NAIP_FOLDER, filename)
                output_path = os.path.join(OUTPUT_FOLDER, "pred_" + filename)
                predict_image(model, input_path, output_path)
                print(n, f"Prediction for {filename} saved to {output_path}")

