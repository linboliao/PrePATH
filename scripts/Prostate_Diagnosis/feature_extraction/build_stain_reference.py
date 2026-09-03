#!/usr/bin/env python
"""Build a stain-normalisation reference from the TRAINING cohort.

The frozen AB_MIL head was trained on RAW virchow2 features from the 迈新 / 省立 /
新昌 training slides. To help the 301 cohort we normalise 301 patches toward the
TRAINING look - not toward the generic colorectal utils/TUM-AEKDYIAK.tif.

Samples patches from N random training slides (all centres, .svs and .kfb),
keeps the tissue-dense ones, and montages the grid*grid patches whose mean
optical density is closest to the training-cohort centroid. That montage is the
Macenko / Reinhard / Vahadane fit target.

out:  <OUT>/stain_reference_train_montage.png
      <OUT>/stain_reference_meta.json
      also copied to  PrePATH/utils/stain_reference_train_montage.png

env:  needs LD_LIBRARY_PATH=<clam-env>/lib for pandas + Aslide (.kfb)
"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import h5py
from PIL import Image

REPO = '/NAS2/Data1/lbliao/Code-195/PrePATH'
sys.path.insert(0, REPO)
DEV_CSV = '/NAS2/Data1/lbliao/Code-195/MIL_BASELINE/datasets/ProstateDiagnosis/dev.csv'
COORDS_DIR = '/NAS145/linboliao/Data/迈新生物_特征/Prostate_Diagnosis/MIL训练数据/patches_0_224/patches'
PSIZE = 224


def open_wsi(p):
    ext = p.rsplit('.', 1)[-1].lower()
    if ext in ('svs', 'tif', 'tiff', 'ndpi', 'mrxs'):
        import openslide
        return 'os', openslide.OpenSlide(p)
    if ext in ('kfb', 'tmap', 'sdpc'):
        from wsi_core.Aslide.aslide import Slide
        return 'as', Slide(p)
    raise ValueError(ext)


def read_patch(kind, h, x, y):
    if kind == 'os':
        return np.array(h.read_region((int(x), int(y)), 0, (PSIZE, PSIZE)).convert('RGB'))
    a = np.array(h.read_fixed_region((int(x), int(y)), 0, (PSIZE, PSIZE)))
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    return a


def tissue_ok(a):
    g = a.mean(2)
    return (g < 220).mean() > 0.55 and g.std() > 12 and (g < 35).mean() < 0.5


def mean_od(a):
    a = a.astype(np.float32) + 1.0
    return (-np.log(a / 256.0)).reshape(-1, 3).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/NAS145/linboliao/Data/迈新生物_特征/Prostate_Diagnosis/MIL外部测试/feat_0_224_stains')
    ap.add_argument('--n-slides', type=int, default=40)
    ap.add_argument('--patches-per-slide', type=int, default=25)
    ap.add_argument('--grid', type=int, default=5)          # grid*grid montage tiles
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import pandas as pd
    dev = pd.read_csv(DEV_CSV)
    cand = dev.sample(min(a.n_slides * 4, len(dev)), random_state=a.seed).to_dict('records')
    rng = random.Random(a.seed)

    collected = []          # (mean_od, patch_rgb, slide_id, x, y, center)
    used = []
    for r in cand:
        if len(used) >= a.n_slides:
            break
        h5p = os.path.join(COORDS_DIR, f"{r['slide_id']}.h5")
        if not os.path.exists(str(r['raw_path'])) or not os.path.exists(h5p):
            continue
        try:
            kind, h = open_wsi(r['raw_path'])
        except Exception as e:
            print('open fail', r['slide_id'], repr(e))
            continue
        with h5py.File(h5p, 'r') as f:
            coords = f['coords'][:]
        pick = rng.sample(range(len(coords)), min(a.patches_per_slide, len(coords)))
        got = 0
        for i in pick:
            x, y = coords[i]
            try:
                patch = read_patch(kind, h, x, y)
            except Exception:
                continue
            if patch.shape[:2] != (PSIZE, PSIZE) or not tissue_ok(patch):
                continue
            collected.append((mean_od(patch), patch, r['slide_id'], int(x), int(y), r['center']))
            got += 1
        try:
            h.close()
        except Exception:
            pass
        if got:
            used.append((r['slide_id'], r['center']))
            print(f"{r['slide_id']} ({r['center']}): +{got} tissue patches   [{len(used)}/{a.n_slides} slides]", flush=True)

    if len(collected) < a.grid * a.grid:
        sys.exit(f'only {len(collected)} tissue patches collected, need {a.grid**2}')

    ods = np.stack([c[0] for c in collected])
    centroid = np.median(ods, 0)
    dist = np.linalg.norm(ods - centroid, axis=1)
    keep = np.argsort(dist)[:a.grid * a.grid]
    tiles = [collected[i][1] for i in keep]

    G = a.grid
    mont = np.zeros((G * PSIZE, G * PSIZE, 3), np.uint8)
    for j, t in enumerate(tiles):
        rr, cc = divmod(j, G)
        mont[rr * PSIZE:(rr + 1) * PSIZE, cc * PSIZE:(cc + 1) * PSIZE] = t
    outp = os.path.join(a.out, 'stain_reference_train_montage.png')
    Image.fromarray(mont).save(outp)
    Image.fromarray(mont).save(os.path.join(REPO, 'utils', 'stain_reference_train_montage.png'))

    from collections import Counter
    meta = {
        'built_from': 'MIL训练数据 (dev.csv)',
        'n_slides_sampled': len(used),
        'centres': dict(Counter(c for _, c in used)),
        'n_tissue_patches_pool': len(collected),
        'montage': f'{G}x{G}',
        'centroid_meanOD': centroid.tolist(),
        'seed': a.seed,
        'tiles': [{'slide': collected[i][2], 'x': collected[i][3], 'y': collected[i][4],
                   'center': collected[i][5]} for i in keep],
    }
    json.dump(meta, open(os.path.join(a.out, 'stain_reference_meta.json'), 'w'), indent=2, ensure_ascii=False)
    print(f'\nmontage -> {outp}   ({len(tiles)} tiles / {len(used)} slides / centres {meta["centres"]})')
    print(f'copy    -> {REPO}/utils/stain_reference_train_montage.png')


if __name__ == '__main__':
    main()
