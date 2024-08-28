import os
import shutil

cities = ['LA', 'BA', 'LB', 'SF', 'SJ']

for city in cities:

    # 定义源文件夹和目标文件夹的路径
    source_folder = f"/workspace/ericyi/Surface Albedo/data/{city}/512/naip_train"
    output_folder = f"/workspace/ericyi/Surface Albedo/data/{city}/572/naip_output"
    destination_folder = f"/workspace/ericyi/Surface Albedo/data/{city}/572/naip_train"

    # 如果目标文件夹不存在，创建它
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if filename.endswith(".tif"):
            # 创建新文件名，基于旧文件名添加'_padded'
            new_filename = filename.replace(".tif", "_padded.tif")

            # 构建完整的源文件和目标文件路径
            source_file_path = os.path.join(output_folder, new_filename)
            destination_file_path = os.path.join(destination_folder, new_filename)

            # 复制文件
            if os.path.exists(source_file_path):
                shutil.copy(source_file_path, destination_file_path)
                print(f"File {new_filename} copied to {destination_folder}.")
            else:
                print(f"File {new_filename} does not exist in {output_folder}.")
