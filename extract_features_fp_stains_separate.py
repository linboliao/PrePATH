import argparse
import glob
import os
import time
from datetime import datetime
from multiprocessing import Process

import numpy as np
import openslide
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_custom_transformer, get_model
from utils.file_utils import save_hdf5, collate_features
from utils.stains import TorchStain
from wsi_core.Aslide.simple import ImgReader
from torchvision import transforms

# 解除图像尺寸限制
Image.MAX_IMAGE_PIXELS = None
import warnings

warnings.filterwarnings('ignore')


def get_file_extensions(directory):
    from pathlib import Path
    extensions = set()
    for path_str in directory:
        for file_path in Path(path_str).rglob('*'):
            if os.path.isfile(file_path):
                ext = file_path.suffix
                if ext:  # 确保文件有扩展名
                    extensions.add(ext.lower())  # 使用小写存储以避免重复
    return ';'.join(sorted(extensions))


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


def save_feature(path, bag_name, features, coords):
    s = time.time()
    for feature, coord in zip(features, coords):
        feat_path = f'{path}/{bag_name}_{coord[0]}_{coord[1]}.pt'
        # print(feature.shape)
        # print(feat_path)
        torch.save(feature.clone(), feat_path)
    e = time.time()
    print('Feature is successfully saved at: {}, cost: {:.1f} s'.format(path, e - s))


def save_hdf5_subprocess(output_path, asset_dict):
    kwargs = {'output_path': output_path, 'asset_dict': asset_dict,
              'attr_dict': None, 'mode': 'w'}
    process = Process(target=save_hdf5, kwargs=kwargs)
    process.start()


def save_feature_subprocess(path, bag_name, features, coords):
    kwargs = {'features': features, 'path': path, 'coords': coords, 'bag_name': bag_name}
    process = Process(target=save_feature, kwargs=kwargs)
    process.start()


def light_compute_w_loader(file_path, wsi, model,
                           batch_size=8, verbose=0, print_every=20, pretrained=True,
                           custom_downsample=1, target_patch_size=-1, custom_transformer=None):
    """
    Do not save features to h5 file to save storage
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding
    """
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, custom_transforms=custom_transformer,
                                 custom_downsample=custom_downsample, target_patch_size=target_patch_size, fast_read=True)
    kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}
    print('Data Loader args:', kwargs)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path, len(loader)))

    features_list = []
    coords_list = []
    _start_time = time.time()
    # cal_time = time.time()
    for count, (batch, coords) in enumerate(loader):
        # read_time_flag = time.time()
        # img_read_time = abs(read_time_flag - cal_time)
        # print('Reading images time:', img_read_time)
        with torch.no_grad():
            if count % print_every == 0:
                batch_time = time.time()
                print('batch {}/{}, {} files processed, used_time: {} s'.format(
                    count, len(loader), count * batch_size, batch_time - _start_time))

            batch = batch.to(device, non_blocking=True)

            features = model(batch)
            features = features.cpu()
            features_list.append(features)
            coords_list.append(coords)
            # cal_time = time.time()
        # print('Calculation time: {} s'.format(cal_time-read_time_flag))
    if len(features_list) > 0:
        features = torch.cat(features_list, dim=0)
        coords = np.concatenate(coords_list, axis=0)
        return features, coords
    else:
        return None, None


def find_all_wsi_paths(wsi_root, extentions):
    """
    find the full wsi path under data_root, return a dict {slide_id: full_path}
    """
    # to support more than one ext, e.g., support .svs and .mrxs
    result = {}
    for ext in extentions.split(';'):
        print('Process format:', ext)
        ext = ext[1:]
        all_paths = glob.glob(os.path.join(wsi_root, '**'), recursive=True)
        all_paths = [i for i in all_paths if i.split('.')[-1].lower() == ext.lower()]
        for h in all_paths:
            slide_name = os.path.split(h)[1]
            slide_id = '.'.join(slide_name.split('.')[0:-1])
            result[slide_id] = h
    print("found {} wsi".format(len(result)))
    return result


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_coors_dir', type=str, default="")
parser.add_argument('--data_slide_dir', type=str, default="")
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--csv_path', type=str, default="")
parser.add_argument('--feat_dir', type=str, default="")
parser.add_argument('--batch_size', type=int, default=96)
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--model', type=str, default='uni')
parser.add_argument('--datatype', type=str)
parser.add_argument('--save_storage', type=str, default='no')
parser.add_argument('--device', type=int, default=0)

parser.add_argument('--ignore_partial', default='yes', type=str)

# Histlogy-pretrained MAE setting
# parser.add_argument('--mae_checkpoint', type=str, default=None, help='path to pretrained mae checkpoint')

args = parser.parse_args()

if __name__ == '__main__':
    # 提取特征，染色归一化，每个patch保留一个特征文件，用于训练癌症去特征分类
    process_start_time = time.time()
    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)

    os.makedirs(args.feat_dir, exist_ok=True)

    print('loading model checkpoint:', args.model)
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:{}, GPU Count:{}'.format(device.type, torch.cuda.device_count()))

    model = get_model(args.model, device, torch.cuda.device_count())

    total = len(bags_dataset)
    print('Total number of WSIs:', total)
    # obtain slide_id
    get_slide_id = lambda idx: str(bags_dataset[idx]).split(args.slide_ext)[0]
    # check the exists wsi
    exist_idxs = []

    all_wsi_paths = find_all_wsi_paths(args.data_slide_dir, args.slide_ext)

    for bag_candidate_idx in range(total):
        slide_id = get_slide_id(bag_candidate_idx)
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_coors_dir, 'patches', bag_name)
        if not os.path.exists(h5_file_path):
            print(h5_file_path, 'does not exist ...')
            continue
        else:
            exist_idxs.append(bag_candidate_idx)

    print('WSIs need to be processed: {} of {}'.format(len(exist_idxs), total))

    for index, bag_candidate_idx in enumerate(exist_idxs):
        slide_id = get_slide_id(bag_candidate_idx)
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_coors_dir, 'patches', bag_name)

        # TCGA
        if slide_id not in all_wsi_paths.keys():
            continue
        slide_file_path = all_wsi_paths[slide_id]
        print('Time:', datetime.now().strftime('"%Y-%m-%d, %H:%M:%S"'))
        print('\nprogress: {}/{}, slide_id: {}'.format(index, len(exist_idxs), slide_id))

        output_h5_path = os.path.join(args.feat_dir, 'h5_files', args.model, bag_name)
        bag_base, _ = os.path.splitext(bag_name)

        one_slide_start = time.time()
        try:
            wsi = get_wsi_handle(slide_file_path)
        except:
            print('Failed to read WSI:', slide_file_path)
            continue

        custom_transformer = transforms.Compose([
            TorchStain(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        features, coords = light_compute_w_loader(h5_file_path, wsi,
                                                  model=model, batch_size=args.batch_size, verbose=1, print_every=20,
                                                  custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size,
                                                  custom_transformer=custom_transformer)

        # save results
        # TODO 以文件名作为 patch 特征标签
        import pandas as pd

        df = pd.read_csv(csv_path)
        label = df[df['slide_id'] == slide_id]['label'].values[0]
        if label == 1:
            save = os.path.join(args.feat_dir, 'TUM')

        else:
            save = os.path.join(args.feat_dir, 'NORM')
        os.makedirs(save, exist_ok=True)
        base = os.path.splitext(bag_name)[0]
        save_feature_subprocess(save, base, features, coords)
        print('feature shape:', features.shape)
        print('coords shape:', coords.shape)

        print('time per slide: {:.1f}'.format(time.time() - one_slide_start))

    print('Time used for this dataset:{:.1f}'.format(time.time() - process_start_time))
    print('Extracting end', end='')
