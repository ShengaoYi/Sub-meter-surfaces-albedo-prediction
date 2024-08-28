import rasterio
import numpy as np
import jenkspy

def read_and_process_raster(input_raster, output_raster):
    with rasterio.open(input_raster) as src:
        # 读取栅格数据
        data = src.read(1)

        # 将所有值乘以100并转换为uint8类型
        data = (data * 100).astype(np.uint8)

        # 使用Jenks自然断点分类法将数据分为五类
        data_flat = data.flatten()
        breaks = jenkspy.jenks_breaks(data_flat, n_classes=5)

        # 创建分类数据
        data_classified = np.digitize(data, bins=breaks, right=True)

        # 更新元数据
        profile = src.profile
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw'
        )

        # 写入输出栅格
        with rasterio.open(output_raster, 'w', **profile) as dst:
            dst.write(data_classified, 1)


# 输入和输出栅格路径
input_raster = '/workspace/ericyi/Surface Albedo/data/Philadelphia/Albedo/Philadelphia_Albedo_UNet.tif'
output_raster = '/workspace/ericyi/Surface Albedo/data/Philadelphia/Albedo/Philadelphia_Albedo_UNet_Categorized.tif'

# 进行处理
read_and_process_raster(input_raster, output_raster)

print(f"Processing completed. Output saved to {output_raster}")
