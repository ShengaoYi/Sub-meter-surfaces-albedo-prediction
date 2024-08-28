import rasterio

def apply_mask(naip_path, mask_path, output_path):
    with rasterio.open(naip_path) as naip_src:
        naip_data = naip_src.read()
        naip_meta = naip_src.meta

        with rasterio.open(mask_path) as mask_src:
            mask_data = mask_src.read(1)  # 读取单波段

            # 对NAIP的每一个波段应用掩膜
            for i in range(naip_data.shape[0]):
                naip_data[i] = naip_data[i] * mask_data

        with rasterio.open(output_path, 'w', **naip_meta) as dst:
            dst.write(naip_data)

naip_file = r"F:\ca037_2016.tif"
single_file = r"E:\Project\Surface Albedo\data\LA\Impervious\Impervious_LA_2016_Single.tif"
output_file = r"G:\NAIP\LA\2016\NAIP_LA_2016\NAIP_LA_2016_Impervious.tif"
apply_mask(naip_file, single_file, output_file)

