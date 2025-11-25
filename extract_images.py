import openslide
import os
import h5py
import numpy as np
from multiprocessing.pool import Pool
import glob
import argparse
from wsi_core.WholeSlideImage import WholeSlideImage
from configs import resolution as RESOLUTION
import openslide


def adjust_size(object_power):
    steps = RESOLUTION.STEPS
    sizes = RESOLUTION.SIZES
    if object_power <= 30:
        return sizes["20x"], steps["20x"]
    elif 30 < object_power <= 60:
        return sizes["40x"], steps["40x"]
    else:
        return sizes["80x"], steps["80x"]


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


def read_images(arg):
    h5_path, save_root, wsi_path, auto_size, level, size = arg
    if wsi_path is None:
        return

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    print('Processing:', h5_path, wsi_path, flush=True)
    try:
        h5 = h5py.File(h5_path)
    except:
        print(f'{h5_path} is not readable....')
        return
        # raise RuntimeError(f'{h5_path} is not readable....')

    _num = len(h5['coords'])
    if _num == len(os.listdir(save_root)):
        return

    coors = h5['coords']

    # Get the WSI handle
    wsi_handle = get_wsi_handle(wsi_path)

    # If auto_size is enabled, determine the appropriate size and level based on the WSI's object power
    if auto_size:
        try:
            WSI_object = WholeSlideImage(wsi_path)
            object_power = WSI_object.object_power
            patch_size, step_size = adjust_size(object_power)
            # Use patch_size as size
            size = patch_size, patch_size
            print(f"Auto-adjusted size to {size} and level to {level} based on object power {object_power} for {os.path.basename(wsi_path)}")
        except Exception as e:
            print(f"Failed to auto-adjust size for {wsi_path}: {e}")
            # Fall back to default values if there's an error
            if not isinstance(size, tuple):
                size = (size, size)
    elif not isinstance(size, tuple):
        size = (size, size)

    for x, y in coors:
        p = os.path.join(save_root, '{}_{}_{}_{}.jpg'.format(x, y, size[0], size[1]))
        if os.path.exists(p):
            continue
        try:
            img = wsi_handle.read_region((x, y), level, size).convert('RGB')
            img.save(p)
        except:
            print('Failed to read: {}, {}, {}'.format(wsi_path, x, y))


def find_all_wsi_paths(wsi_root, extentions):
    """
    find the full wsi path under data_root, return a dict {slide_id: full_path}
    """
    # to support more than one ext, e.g., support .svs and .mrxs
    result = {}

    all_paths = glob.glob(os.path.join(wsi_root, '**'), recursive=True)

    for ext in extentions.split(';'):
        print('Process format:', ext)
        paths = [i for i in all_paths if i.split('.')[-1].lower() == ext.lower()]
        for h in paths:
            slide_name = os.path.split(h)[1]
            slide_id = '.'.join(slide_name.split('.')[0:-1])
            result[slide_id] = h
    print("found {} wsi".format(len(result)))
    return result


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datatype')
    parser.add_argument('--wsi_format')
    parser.add_argument('--level', type=int, default=0, help='Default level, used if auto_size is disabled')
    parser.add_argument('--size', type=int, default=512, help='Default size, used if auto_size is disabled')
    parser.add_argument('--auto_size', action='store_true', help='Use adjust_size to automatically determine size and level based on each WSI')
    parser.add_argument('--cpu_cores', type=int, default=48)
    parser.add_argument('--h5_root')
    parser.add_argument('--save_root')
    parser.add_argument('--wsi_root')
    return parser


if __name__ == '__main__':
    parser = argparser().parse_args()

    datatype = parser.datatype
    wsi_format = parser.wsi_format
    auto_size = parser.auto_size
    level = parser.level
    size = parser.size

    h5_root = parser.h5_root
    save_root = parser.save_root
    wsi_root = parser.wsi_root
    all_wsi_paths = find_all_wsi_paths(parser.wsi_root, parser.wsi_format)
    h5_paths, wsi_paths, save_roots = [], [], []
    for slide_id, path in all_wsi_paths.items():
        h5_path = os.path.join(h5_root, f'{slide_id}.h5')

        if os.path.exists(h5_path):
            h5_paths.append(h5_path)
            wsi_paths.append(path)
            save_roots.append(os.path.join(save_root, slide_id))

    # Include auto_size flag in the arguments
    args = [(h5, sr, wsi_path, auto_size, level, size) for h5, wsi_path, sr in zip(h5_paths, wsi_paths, save_roots)]

    mp = Pool(parser.cpu_cores)
    mp.map(read_images, args)
    print('All slides have been cropped!')
