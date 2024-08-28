import os
import rasterio
from rasterio.merge import merge
from rasterio.windows import Window

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

def crop_and_save_image(src_image, output_file, crop_margin=42):
    with rasterio.open(src_image) as src:
        window = Window(crop_margin, crop_margin, src.width - 2 * crop_margin, src.height - 2 * crop_margin)
        cropped_data = src.read(window=window)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": window.height,
            "width": window.width,
            "transform": rasterio.windows.transform(window, src.transform)
        })
        with rasterio.open(output_file, "w", **out_meta) as dst:
            dst.write(cropped_data)

for city in cities:

    # 指定文件夹路径
    impervious_path = fr"/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Albedo_UNet_Masked_1024_new"
    pervious_path = fr"/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Pervious_Albedo_UNet_Masked_1024_new"
    output_dir = fr"/workspace/ericyi/Surface Albedo/data/{city}/Albedo/Albedo_UNet_Masked_512_Full_new"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取impervious文件夹中的所有TIFF文件
    impervious_tiff_files = [os.path.join(impervious_path, filename) for filename in os.listdir(impervious_path) if
                             filename.endswith('.tif')]

    n = 0
    # 遍历每个impervious TIFF文件，找到相应的pervious TIFF文件，并进行处理
    for impervious_file in impervious_tiff_files:
        filename = os.path.basename(impervious_file)
        pervious_file = os.path.join(pervious_path, filename)

        if os.path.exists(pervious_file):
            n += 1
            # 裁剪并保存
            cropped_output_file = os.path.join(output_dir, filename)
            if os.path.exists(cropped_output_file):
                continue
            # 读取文件并合并
            with rasterio.open(impervious_file) as impervious_src, rasterio.open(pervious_file) as pervious_src:
                src_files_to_mosaic = [impervious_src, pervious_src]
                mosaic, out_trans = merge(src_files_to_mosaic)
                mosaic_output_file = os.path.join(output_dir, filename)

                # 保存合并后的影像
                with rasterio.open(mosaic_output_file, 'w', driver='GTiff',
                                   height=mosaic.shape[1], width=mosaic.shape[2],
                                   count=src_files_to_mosaic[0].count,
                                   dtype=mosaic.dtype,
                                   crs=src_files_to_mosaic[0].crs,
                                   transform=out_trans) as mosaic_dst:
                    mosaic_dst.write(mosaic)


                crop_and_save_image(mosaic_output_file, cropped_output_file, crop_margin=256)
            print(n, pervious_file)

