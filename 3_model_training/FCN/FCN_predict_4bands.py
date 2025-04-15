import os
import torch
import torch.nn as nn
import rasterio
from FCN import FCN

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    model = FCN(input_channels=4, output_channels=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_image(model, input_path, output_path):
    with rasterio.open(input_path) as src:
        naip_image = src.read().astype('float32').transpose((1, 2, 0))
        transform = src.transform  # 获取转换信息
        crs = src.crs  # 获取坐标参考系统

    naip_image[(naip_image == 256).all(axis=2)] = 0

    naip_tensor = torch.tensor(naip_image.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(naip_tensor)
        output_image = output.cpu().squeeze(0).squeeze(0).numpy()

    output_image[output_image < 0] = 0

    with rasterio.open(output_path, 'w', driver='GTiff', height=output_image.shape[0], width=output_image.shape[1],
                       count=1, dtype=output_image.dtype, crs=crs, transform=transform) as dst:
        dst.write(output_image, 1)

if __name__ == "__main__":
    MODEL_PATH = "/workspace/ericyi/Surface Albedo/models/FCN/FCN_epoch_200_all_cities_512.pth"
    INPUT_FOLDER = r"/workspace/ericyi/Surface Albedo/data/SJ/512/naip_test"
    OUTPUT_FOLDER = r"/workspace/ericyi/Surface Albedo/data/SJ/512/naip_test_Albedo_FCN_512_output_200"

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    model = load_model(MODEL_PATH)

    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".tif"):
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, "pred_" + filename)
            predict_image(model, input_path, output_path)
            print(f"Prediction for {filename} saved to {output_path}")
