import os
import rasterio

citys = ['LA', 'BA', 'SF', 'LB', 'SJ']

base_dir = r'/workspace/ericyi/Surface Albedo/data'

def remove_mismatched_images(naip_dir, roof_dir):
    for filename in os.listdir(naip_dir):
        naip_filepath = os.path.join(naip_dir, filename)
        roof_filepath = os.path.join(roof_dir, filename)

        # 检查文件是否存在
        if not os.path.exists(roof_filepath):
            continue

        # 读取NAIP图像的尺寸
        with rasterio.open(naip_filepath) as src:
            width, height = src.width, src.height

        # 检查尺寸
        if width != 572 or height != 572:
            os.remove(naip_filepath)
            os.remove(roof_filepath)
            print(f"Removed {filename} from both directories due to mismatched size.")

for city in citys:
    naip_dir = os.path.join(base_dir, city, '572', f'naip_test')
    roof_dir = os.path.join(base_dir, city, '572', f'roof_test')
    print(naip_dir)

    remove_mismatched_images(naip_dir, roof_dir)


