import os

import concurrent.futures
import traceback
from datetime import datetime

from torch.utils.data import DataLoader, Dataset
from models import get_model, get_custom_transformer

import argparse
from multiprocessing import Process

import torch
from PIL import Image

import torchvision.transforms as transforms
from tqdm import tqdm
import multiprocessing as mp
import time

from utils.stains import TorchStain

"""
    从NCT 图片上基于UNI提取特征
    标签癌、非癌
"""


class ImageDataset(Dataset):
    def __init__(self, data_root, feat_dir, transform=None, mode=None, train_ratio=0.8, preload=True, num_workers=None):
        self.data_root = data_root
        self.transform = transform
        self.mode = mode
        self.train_ratio = train_ratio
        self.preload = preload
        self.num_workers = num_workers if num_workers is not None else mp.cpu_count() * 4

        self.classes = [d.name for d in os.scandir(data_root) if d.is_dir()]
        self.classes.sort()

        self.image_paths = []
        self.feat_paths = []
        for class_name in self.classes:
            class_dir = os.path.join(data_root, class_name)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    base, ext = os.path.splitext(img_file)
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                        self.image_paths.append(os.path.join(class_dir, img_file))
                        if class_name == 'TUM':
                            self.feat_paths.append(os.path.join(feat_dir, f'TUM/{base}.pt'))
                        else:
                            self.feat_paths.append(os.path.join(feat_dir, f'NORM/{base}.pt'))

        num_samples = len(self.image_paths)
        num_train = int(num_samples * train_ratio)
        indices = torch.randperm(num_samples).tolist()
        if not mode:
            self.indices = indices
        elif mode == 'train':
            self.indices = indices[:num_train]
        else:
            self.indices = indices[num_train:]

        self.preloaded_data = []
        if self.preload:
            self._parallel_preload()

    def _load_single_image(self, idx):
        """加载单个图像（用于并行处理）"""
        img_path = self.image_paths[idx]
        feat_path = self.feat_paths[idx]
        image = Image.open(img_path).convert('RGB')
        return image, feat_path

    def _parallel_preload(self):
        """并行预加载数据并显示进度条"""
        print(f"开始并行预加载 {len(self.indices)} 张图像，使用 {self.num_workers} 个进程...")
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 按排序顺序提交任务
            future_to_idx = {}
            for idx in self.indices:
                future = executor.submit(self._load_single_image, idx)
                future_to_idx[future] = idx
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(future_to_idx), desc="预加载进度"):
                idx = future_to_idx[future]
                try:
                    image, feat_path = future.result()
                    self.preloaded_data.append((image, feat_path))
                except Exception as e:
                    traceback.print_exc()
                    print(f"加载索引 {idx} 失败: {e}")

        end_time = time.time()
        print(f"预加载完成! 耗时: {end_time - start_time:.2f} 秒")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.preload:
            image, feat_path = self.preloaded_data[idx]
        else:
            image, feat_path = self._load_single_image(idx)

        if self.transform:
            image = self.transform(image)

        return image, feat_path

    def get_class_names(self):
        """返回类别名称列表"""
        return self.classes


def save_feature(paths, features):
    s = time.time()
    for feature, path in zip(features, paths):
        torch.save(feature.clone(), path)
    e = time.time()
    print('Feature is successfully saved, cost: {:.1f} s'.format(e - s))


def save_feature_subprocess(feature, path):
    kwargs = {'features': feature, 'paths': path}
    process = Process(target=save_feature, kwargs=kwargs)
    process.start()


def light_compute_w_loader(loader, model, print_every=20):
    features_list = []
    paths_list = []
    _start_time = time.time()
    for count, (batch, path) in enumerate(loader):
        with torch.no_grad():
            if count % print_every == 0:
                batch_time = time.time()
                print('batch {}/{}, {} files processed, used_time: {} s'.format(
                    count, len(loader), count * len(batch), batch_time - _start_time))

            batch = batch.to(device, non_blocking=True)
            features = model(batch)
            features = features.cpu()
            features_list.append(features)
            paths_list.append(path)
    features = torch.cat(features_list, dim=0)
    paths = [item for sublist in paths_list for item in sublist]
    return features, paths


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--patch_img_dir', type=str, default='')
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--mode', type=str, default=None)
parser.add_argument('--model', type=str)
parser.add_argument('--datatype', type=str)

args = parser.parse_args()

if __name__ == '__main__':
    process_start_time = time.time()
    os.makedirs(os.path.join(args.feat_dir, 'TUM'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'NORM'), exist_ok=True)

    print('loading model checkpoint:', args.model)
    device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:{}, GPU Count:{}'.format(device.type, torch.cuda.device_count()))
    print('Total number of Images:')
    print('Time:', datetime.now().strftime('"%Y-%m-%d, %H:%M:%S"'))

    # custom_transformer = transforms.Compose([TorchStain()] + get_custom_transformer(args.model).transforms)
    custom_transformer = get_custom_transformer(args.model)


    def collate_features(batch):
        img = torch.stack([item[0] for item in batch], dim=0)
        path = [item[1] for item in batch]
        assert len(img.shape) == 4, "img shape is wrong, please check"
        return [img, path]


    dataset = ImageDataset(args.patch_img_dir, args.feat_dir, custom_transformer, mode=args.mode)
    kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, **kwargs, collate_fn=collate_features)

    model = get_model(args.model, device, torch.cuda.device_count())
    features, paths = light_compute_w_loader(dataloader, model=model)

    save_feature_subprocess(features, paths)

    print('Time used for this dataset:{:.1f}'.format(time.time() - process_start_time))
    print('Extracting end', end='')
