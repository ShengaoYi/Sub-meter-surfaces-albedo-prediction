import rasterio

city = 'Miami'

input_file = fr'/workspace/ericyi/Surface Albedo/data/NewYorkCity/NAIP4/NAIP_NYC_2022_26918.tif'  # 输入文件的路径
output_file = fr'/workspace/ericyi/Surface Albedo/data/NewYorkCity/NAIP_NYC_2022_26918_compressed.tif'  # 输出文件的路径

# 读取原始的TIFF文件
with rasterio.open(input_file) as src:
    data = src.read()  # 读取数据
    out_meta = src.meta  # 读取元数据

# 更新元数据以包括LZW压缩
out_meta.update({
    'compress': 'lzw',
'BIGTIFF': 'YES'
})

# 将数据和更新后的元数据写入到新的TIFF文件中
with rasterio.open(output_file, 'w', **out_meta) as dest:
    dest.write(data)