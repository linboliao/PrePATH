import argparse
import json
import os

import h5py
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, Point
from pathlib import Path
import openslide

from wsi_core.Aslide.simple import ImgReader


def get_wsi_handle(wsi_path):
    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f'{wsi_path} is not found')
    postfix = wsi_path.split('.')[-1]
    if postfix.lower() in ['svs', 'tif', 'ndpi', 'tiff', 'mrxs']:
        handle = openslide.OpenSlide(wsi_path)
    elif postfix.lower() in ['jpg', 'jpeg', 'tiff', 'png']:
        handle = ImgReader(wsi_path)

    elif postfix.lower() in ['kfb', 'tmap', 'sdpc']:
        from wsi_core.Aslide.aslide import Slide
        handle = Slide(wsi_path)
    else:
        raise NotImplementedError(f'{postfix} is not implemented...')
    return handle


def extract_patch_coordinates_from_geojson(geojson_path, patch_size=244):
    """
    从GeoJSON文件中提取所有轮廓内部patch的左上角坐标

    参数:
        geojson_path: GeoJSON文件路径
        patch_size: patch大小，默认244px

    返回:
        list: 包含所有patch左上角坐标的列表 [(x1, y1), (x2, y2), ...]
    """
    # 读取GeoJSON文件
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)

    all_patch_coordinates = []

    # 遍历GeoJSON中的所有要素
    for feature in geojson_data['features']:
        geometry = feature['geometry']

        # 处理多边形要素
        if geometry['type'] == 'Polygon':
            # 提取多边形坐标
            polygon_coords = geometry['coordinates'][0]  # 主多边形坐标
            polygon = Polygon(polygon_coords)

            # 计算多边形的边界框
            min_x, min_y, max_x, max_y = polygon.bounds

            # 在边界框内生成patch网格
            for x in np.arange(min_x, max_x, patch_size):
                for y in np.arange(min_y, max_y, patch_size):

                    # 检查patch中心是否在多边形内
                    center_x = x + patch_size / 2
                    center_y = y + patch_size / 2
                    center_point = Point(center_x, center_y)

                    if polygon.contains(center_point):
                        all_patch_coordinates.append((int(x), int(y)))

        # 处理多重多边形要素
        elif geometry['type'] == 'MultiPolygon':
            for polygon_coords in geometry['coordinates']:
                polygon = Polygon(polygon_coords[0])  # 主多边形

                # 计算多边形的边界框
                min_x, min_y, max_x, max_y = polygon.bounds

                # 在边界框内生成patch网格
                for x in np.arange(min_x, max_x, patch_size):
                    for y in np.arange(min_y, max_y, patch_size):
                        # 检查patch中心是否在多边形内
                        center_x = x + patch_size / 2
                        center_y = y + patch_size / 2
                        center_point = Point(center_x, center_y)

                        if polygon.contains(center_point):
                            all_patch_coordinates.append((int(x), int(y)))

    return all_patch_coordinates


def save_coordinates_to_h5(coordinates, output_path):
    """
    将坐标保存到HDF5文件中

    参数:
        coordinates: 坐标列表
        output_path: 输出H5文件路径
    """
    # 转换为numpy数组
    coords_array = np.array(coordinates, dtype=np.int32)

    # 保存到HDF5文件
    with h5py.File(output_path, 'w') as h5f:
        # 创建数据集
        h5f.create_dataset('coords',
                           data=coords_array,
                           compression='gzip',  # 使用压缩减少文件大小
                           compression_opts=9)

        # 添加元数据
        h5f.attrs['patch_size'] = 244
        h5f.attrs['number_of_patches'] = len(coordinates)
        h5f.attrs['description'] = '左上角坐标列表 (x, y)'


# 主函数
def main(geojson_path, output_h5_path, patch_size=244):
    """
    主处理函数

    参数:
        wsi_path: WSI文件路径
        geojson_path: GeoJSON标注文件路径
        output_h5_path: 输出H5文件路径
        patch_size: patch大小
    """
    print(f"开始处理: {geojson_path}")

    # 提取patch坐标
    patch_coordinates = extract_patch_coordinates_from_geojson(
        geojson_path, patch_size
    )

    print(f"找到 {len(patch_coordinates)} 个patch坐标")

    # 保存到HDF5文件
    save_coordinates_to_h5(patch_coordinates, output_h5_path)

    print(f"坐标已保存到: {output_h5_path}")

    return patch_coordinates


def visualize_patches(wsi_path, coordinates, output_image_path, patch_size=244, max_size=1000, alpha=80):
    """
    在WSI缩略图上可视化提取的patch位置
    """
    slide = get_wsi_handle(wsi_path)

    # 生成缩略图
    # thumb = slide.get_thumbnail((max_size, max_size))
    level = len(slide.level_dimensions)
    thumb = slide.read_region((0, 0), level - 1, slide.level_dimensions[level - 1])
    thumb.save(output_image_path.replace('/masks/', '/slides/'))
    # 转换为RGBA模式以支持透明度
    thumb = thumb.convert('RGBA')

    # 创建临时透明图层用于绘制矩形
    overlay = Image.new('RGBA', thumb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 计算缩放比例
    scale_x = thumb.size[0] / slide.dimensions[0]
    scale_y = thumb.size[1] / slide.dimensions[1]

    # 在透明图层上绘制patch位置
    for x, y in coordinates:
        rect_x = int(x * scale_x)
        rect_y = int(y * scale_y)
        rect_w = int(patch_size * scale_x)
        rect_h = int(patch_size * scale_y)

        # 绘制半透明矩形
        draw.rectangle([rect_x, rect_y, rect_x + rect_w, rect_y + rect_h], fill=(255, 0, 0, alpha))

    # 将透明图层与原始图像合并
    result = Image.alpha_composite(thumb, overlay)

    # 保存为PNG格式以保留透明度
    if not output_image_path.lower().endswith('.png'):
        output_image_path = output_image_path + '.png'

    result.save(output_image_path, 'PNG')
    print(f"可视化结果已保存到: {output_image_path}")


parser = argparse.ArgumentParser(description='create patches')
parser.add_argument('--wsi_dir', type=str, default=r'')
parser.add_argument('--label_dir', type=str, default=r'')
parser.add_argument('--save_dir', type=str, default=r'/')
parser.add_argument('--slide_ext', type=str, default=r'')
parser.add_argument('--vis', default=False, action='store_true')

if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(os.path.join(args.save_dir, 'patches'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'slides'), exist_ok=True)

    wsi_dir = Path(args.wsi_dir)
    wsi_paths = []
    for ext in args.slide_ext.split(';'):
        wsi_paths += list(wsi_dir.rglob(f'*.{ext}'))

    for wsi_path in wsi_paths:
        base = wsi_path.stem
        geojson_path = os.path.join(args.label_dir, f'{base}.geojson')
        if not os.path.exists(geojson_path):
            geojson_path = os.path.join(args.label_dir, f'{base.replace("-", " ")}.geojson')
            if not os.path.exists(geojson_path):
                print(f'{base}找不到对应的geojson')
                continue
        h5_path = os.path.join(args.save_dir, f'patches/{base}.h5')
        if os.path.exists(h5_path):
            print(f'{base}已经处理')
            continue
        mask_path = os.path.join(args.save_dir, f'masks/{base}.jpg')
        coords = main(geojson_path, h5_path, 512)
        if args.vis:
            visualize_patches(str(wsi_path), coords, mask_path)
