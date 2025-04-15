import os
import rasterio

# 定义输入和输出文件夹路径
input_folder = "/workspace/ericyi/Surface Albedo/data/Philadelphia/Albedo/Albedo_UNet_1024_Clipped"
output_folder = "/workspace/ericyi/Surface Albedo/data/Philadelphia/Albedo/Albedo_UNet_1024_Masked"
mask_folder = "/workspace/ericyi/Surface Albedo/data/Philadelphia/512_Impervious_UNet_building_mask"

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有.tif文件
for filename in os.listdir(input_folder):
    if filename.endswith(".tif"):
        src_file_path = os.path.join(input_folder, filename)

        mask_filename = "impervious_" + filename[5:]
        mask_path = os.path.join(mask_folder, mask_filename)

        # 读取对应的掩模
        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1)

        dst_file_path = os.path.join(output_folder, filename)

        # 读取图像并应用掩膜
        with rasterio.open(src_file_path) as src:
            image = src.read(1)  # 假设图像是单波段
            # 应用掩膜
            masked_image = image * mask

            # 复制源文件的元数据并更新
            out_meta = src.meta.copy()
            out_meta.update({
                "height": masked_image.shape[0],
                "width": masked_image.shape[1]
            })

            # 保存处理后的图像
            with rasterio.open(dst_file_path, "w", **out_meta) as dst:
                dst.write(masked_image, 1)

        print(f"Masked image saved to {dst_file_path}")
