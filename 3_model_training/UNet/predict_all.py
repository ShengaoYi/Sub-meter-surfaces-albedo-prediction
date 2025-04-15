import os
import torch
import rasterio
import torch.nn as nn
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

def predict_image(model, input_path, output_path):
    with rasterio.open(input_path) as src:
        naip_image = src.read().astype('float32')
        transform = src.transform
        crs = src.crs

    naip_tensor = torch.tensor(naip_image, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(naip_tensor)
        output_image = output.cpu().squeeze(0).squeeze(0).numpy()

    output_image[output_image < 0] = 0

    with rasterio.open(output_path, 'w', driver='GTiff', height=output_image.shape[0], width=output_image.shape[1],
                       count=1, dtype=output_image.dtype, crs=crs, transform=transform) as dst:
        dst.write(output_image, 1)

if __name__ == "__main__":
    MODEL_PATH = "/workspace/ericyi/Surface Albedo/models/UNet/UNet_epoch_50_all_cities_512.pth"
    INPUT_FOLDER = r"/workspace/ericyi/Surface Albedo/data/SJ/512/naip_test"
    OUTPUT_FOLDER = r"/workspace/ericyi/Surface Albedo/data/SJ/512/naip_test_Albedo_UNet_512_output_50"

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    model = load_model(MODEL_PATH)

    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".tif"):
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, "pred_" + filename)
            predict_image(model, input_path, output_path)
            print(f"Prediction for {filename} saved to {output_path}")
