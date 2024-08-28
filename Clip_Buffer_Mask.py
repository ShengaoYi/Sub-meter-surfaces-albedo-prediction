import os
import rasterio
from rasterio.windows import Window


def crop_image(src_path, dst_path, crop_margin, mask_folder):
    # 构造掩模文件的路径
    mask_filename = "impervious_" + os.path.basename(src_path)[5:]
    mask_path = os.path.join(mask_folder, mask_filename)

    # 读取对应的掩模
    with rasterio.open(mask_path) as mask_src:
        mask = mask_src.read(1)

    with rasterio.open(src_path) as src:
        # 计算新的窗口位置和大小
        window = Window(crop_margin, crop_margin, src.width - 2 * crop_margin, src.height - 2 * crop_margin)
        # 使用窗口读取数据
        cropped_data = src.read(window=window)

        # 再次应用掩模，以确保仅impervious区域的预测值被保留
        cropped_data *= mask

        # 更新元数据以匹配裁剪后的尺寸
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": window.height,
            "width": window.width,
            "transform": rasterio.windows.transform(window, src.transform)
        })

        # 写入裁剪后的图像
        with rasterio.open(dst_path, "w", **out_meta) as dst:
            dst.write(cropped_data)


# 定义输入和输出文件夹路径
input_folder = "/workspace/ericyi/Surface Albedo/data/Philadelphia/Albedo/Albedo_UNet_1024"
output_folder = "/workspace/ericyi/Surface Albedo/data/Philadelphia/Albedo/Albedo_UNet_1024_Clipped2"
MASK_FOLDER = fr"/workspace/ericyi/Surface Albedo/data/Philadelphia/512_Impervious_UNet_building_mask"

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 要裁剪的像素大小
crop_margin = 256

# 遍历输入文件夹中的所有.tif文件
for filename in os.listdir(input_folder):
    if filename.endswith(".tif"):
        src_file_path = os.path.join(input_folder, filename)
        dst_file_path = os.path.join(output_folder, filename)
        crop_image(src_file_path, dst_file_path, crop_margin, MASK_FOLDER)
        print(f"Cropped image saved to {dst_file_path}")

