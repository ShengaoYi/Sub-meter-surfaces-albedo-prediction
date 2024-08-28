import os
import shutil

# 指定包含原始Landsat7 TIF文件的文件夹路径
landsat_dir = "/workspace/ericyi/Surface Albedo/data/Landsat7"
# 指定存放城市文件夹的根目录路径
root_dir = "/workspace/ericyi/Surface Albedo/data"

# 检查Landsat7文件夹是否存在
if not os.path.exists(landsat_dir):
    print(f"Landsat7 directory does not exist: {landsat_dir}")
else:
    # 遍历Landsat7文件夹中的每个TIF文件
    for file in os.listdir(landsat_dir):
        if file.endswith(".tif"):
            # 解析城市名称（假设文件名格式为"Landsat7_CITYNAME_其他信息.tif"）
            parts = file.split("_")
            if len(parts) >= 3:
                city_name = parts[1]
                # 构建目标文件夹路径
                target_dir = os.path.join(root_dir, city_name, "Landsat7")
                # 如果目标文件夹不存在，则创建它
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                    print(f"Created directory: {target_dir}")
                # 移动文件
                src_path = os.path.join(landsat_dir, file)
                dst_path = os.path.join(target_dir, file)
                shutil.move(src_path, dst_path)
                print(f"Moved {file} to {target_dir}")

