import os
import torch
import torch.nn as nn
import rasterio
from ResNet import ResNetRegressor

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    model = ResNetRegressor().to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=[3, 4])

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_image(model, input_path, output_path):
    with rasterio.open(input_path) as src:
        naip_image = src.read().transpose((1, 2, 0))
        transform = src.transform  # 获取转换信息
        crs = src.crs  # 获取坐标参考系统

    naip_tensor = torch.tensor(naip_image.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(naip_tensor)
        output_image = output.cpu().squeeze(0).squeeze(0).numpy()

    with rasterio.open(output_path, 'w', driver='GTiff', height=output_image.shape[0], width=output_image.shape[1],
                       count=1, dtype=output_image.dtype, crs=crs, transform=transform) as dst:
        dst.write(output_image, 1)

if __name__ == "__main__":
    MODEL_PATH = "/data/Yi/Surface Albedo/codes/model/resnet_model_512.pth"
    INPUT_FOLDER = r"/data/Yi/Surface Albedo/data/LA/512/naip_test"
    OUTPUT_FOLDER = r"/data/Yi/Surface Albedo/data/LA/512/ResNet_output_100"

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    model = load_model(MODEL_PATH)

    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".tif"):
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, "pred_" + filename)
            predict_image(model, input_path, output_path)
            print(f"Prediction for {filename} saved to {output_path}")
