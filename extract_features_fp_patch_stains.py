import os
import concurrent.futures
import traceback
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models import get_model, get_custom_transformer
import argparse
from multiprocessing import Process
import torch
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
import time

from utils.stains import TorchStain


class ImageDataset(Dataset):
    def __init__(self, image_list, transform=None, preload=True, num_workers=None):
        """
        image_list: 包含 (img_path, save_feat_path) 的列表
        """
        self.image_list = image_list
        self.transform = transform
        self.preload = preload
        self.num_workers = num_workers if num_workers is not None else mp.cpu_count()

        self.preloaded_data = []
        if self.preload:
            self._parallel_preload()

    def _load_single_image(self, idx):
        img_path, feat_path = self.image_list[idx]
        image = Image.open(img_path).convert('RGB')
        return image, feat_path

    def _parallel_preload(self):
        print(f"正在预加载 {len(self.image_list)} 张图像...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_idx = {executor.submit(self._load_single_image, i): i for i in range(len(self.image_list))}
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(future_to_idx), desc="预加载"):
                try:
                    self.preloaded_data.append(future.result())
                except Exception as e:
                    print(f"加载失败: {e}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if self.preload:
            image, feat_path = self.preloaded_data[idx]
        else:
            image, feat_path = self._load_single_image(idx)

        if self.transform:
            image = self.transform(image)
        return image, feat_path


def save_feature(paths, features):
    s = time.time()
    for feature, path in zip(features, paths):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(feature.clone().cpu(), path)
    e = time.time()
    print('Feature is successfully saved, cost: {:.1f} s'.format(e - s))


def save_feature_subprocess(features, paths):
    kwargs = {'features': features, 'paths': paths}
    process = Process(target=save_feature, kwargs=kwargs)
    process.start()
    return process


def light_compute_w_loader(loader, model, device, print_every=20):
    features_list = []
    paths_list = []
    _start_time = time.time()

    model.eval()
    for count, (batch, path) in enumerate(loader):
        with torch.no_grad():
            if count % print_every == 0:
                print('batch {}/{}, {} files processed, used_time: {:.1f} s'.format(
                    count, len(loader), count * len(batch), time.time() - _start_time))

            batch = batch.to(device, non_blocking=True)
            features = model(batch).cpu()
            features_list.append(features)
            paths_list.append(path)

    features = torch.cat(features_list, dim=0)
    paths = [item for sublist in paths_list for item in sublist]
    return features, paths


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--patch_img_dir', type=str, required=True)
parser.add_argument('--feat_dir', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--model', type=str, required=True)

if __name__ == '__main__':
    args = parser.parse_args()
    process_start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    splits = {'train': [], 'val': [], 'test': []}
    classes = sorted([d.name for d in os.scandir(args.patch_img_dir) if d.is_dir()])

    for cls in classes:
        cls_dir = os.path.join(args.patch_img_dir, cls)
        imgs = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

        torch.manual_seed(42)
        indices = torch.randperm(len(imgs)).tolist()

        n = len(imgs)
        t_sep = int(n * 0.7)
        v_sep = int(n * 0.85)

        for i, idx in enumerate(indices):
            img_name = imgs[idx]
            base = os.path.splitext(img_name)[0]
            src = os.path.join(cls_dir, img_name)

            if i < t_sep:
                mode = 'train'
            elif i < v_sep:
                mode = 'val'
            else:
                mode = 'test'

            dst = os.path.join(args.feat_dir, mode, cls, f"{base}.pt")
            splits[mode].append((src, dst))

    model = get_model(args.model, device, torch.cuda.device_count())
    custom_transformer = transforms.Compose([TorchStain('macenko')] + get_custom_transformer(args.model).transforms)


    def collate_features(batch):
        img = torch.stack([item[0] for item in batch], dim=0)
        path = [item[1] for item in batch]
        return [img, path]


    save_procs = []
    for mode in ['train', 'val', 'test']:
        if not splits[mode]: continue

        print(f"\n--- 开始处理 {mode} 集合 (样本数: {len(splits[mode])}) ---")
        dataset = ImageDataset(splits[mode], transform=custom_transformer, preload=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, collate_fn=collate_features)

        features, paths = light_compute_w_loader(dataloader, model, device)

        p = save_feature_subprocess(features, paths)
        save_procs.append(p)

    for p in save_procs:
        p.join()

    print(f'\n所有数据集提取完毕，总耗时: {time.time() - process_start_time:.1f}s')
