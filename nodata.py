import os
import rasterio

def is_empty_tif(file_path):
    """ 检查tif文件是否为空（所有像素值均为0或nodata） """
    try:
        with rasterio.open(file_path) as src:
            nodata = src.nodata
            for _, window in src.block_windows(1):
                data = src.read(1, window=window)
                if nodata is not None:
                    # 检查所有像素是否为0或nodata
                    if not (data == 0).all() and not (data == nodata).all():
                        return False
                else:
                    # 如果没有nodata值，只检查是否所有像素为0
                    if data.any():
                        return False
            return True
    except rasterio.errors.RasterioIOError:
        print(f"无法打开文件：{file_path}")
        return False  # 如果无法打开文件，假设它不为空
root = r'/mnt/shengao/Surface Albedo/data/LA/512/naip_output'

files = os.listdir(root)
i = 0
for file in files:
    path = os.path.join(root, file)
    if is_empty_tif(path):
        i += 1
        os.remove(path)
        print(i, path)
        TreeCanopy_path = os.path.join(r'/mnt/shengao/Surface Albedo/data/LA/512/roof_output', file)
        try:
            os.remove(TreeCanopy_path)
        except:
            continue