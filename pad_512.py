import os
import rasterio
import numpy as np


def pad_to_512x512(input_file, output_file):
    with rasterio.open(input_file) as src:
        data = src.read()
        transform = src.transform
        profile = src.profile

        height, width = data.shape[1], data.shape[2]

        # 如果已经是512x512，直接复制
        if height == 512 and width == 512:
            return

        # 创建一个512x512xnum_bands的零数组
        padded_data = np.zeros((data.shape[0], 512, 512), dtype=data.dtype)

        # 把原始数据复制到零数组中
        padded_data[:, :height, :width] = data

        # 更新profile
        profile.update(width=512, height=512)

        # 写入新文件
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(padded_data)


def main(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".tif"):
            input_path = os.path.join(folder_path, file_name)
            output_path = os.path.join(folder_path, "padded_" + file_name)

            with rasterio.open(input_path) as src:
                width, height = src.width, src.height

                if width != 512 or height != 512:
                    print(f"Padding file: {file_name}")
                    pad_to_512x512(input_path, output_path)


if __name__ == '__main__':
    folder_path = r"F:\Ericyi\Surface Albedo\data\PA\NAIP\NAIP_PA_83_NoVegetation"  # 请替换为你的文件夹路径
    main(folder_path)
