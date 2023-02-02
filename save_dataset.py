import json
import os
import sys
import time
import random
import string
import argparse
from collections import namedtuple
import torch
import torch.utils.data

import numpy as np
from torchvision import transforms

from fonts_datasets import FontsDataset
import font_transforms as font_t

# parOptions = namedtuple('parOptions', ['DP', 'DDP', 'HVD'])  # NU
# parOptions.__new__.__defaults__ = (False,) * len(parOptions._fields)  # NU

# pO = None  # NU
# OnceExecWorker = None  # NU


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def WrkSeeder(_):
    return np.random.seed((torch.initial_seed()) % (2 ** 32))


# def rSeed(sd):  # NU
#     random.seed(sd)
#     np.random.seed(sd)
#     torch.manual_seed(sd)
#     torch.cuda.manual_seed(sd)


# def sum_dict(dst, src):  # NU
#     for key, val in src.items():
#         if key in dst:
#             dst[key] += val
#         else:
#             dst[key] = val
#     return dst
#
#
# def go_through_time(dataloader):  # NU
#     tot_time = {}
#     count = 0
#     for i, times in enumerate(dataloader):
#         for t in times: sum_dict(tot_time, t)
#         count += len(times)
#         print(f'Samples {i}/{len(dataloader)}')
#         for key, val in tot_time.items():
#             print(f'  - {key}: {val / count:.04f}s')


def go_through(dataloader):
    for i, _ in enumerate(dataloader):
        print(f'Samples {i}/{len(dataloader)}')


def save_db(dataset, font_path, args):
    basename = os.path.basename(font_path)
    dst_dir = os.path.splitext(basename)[0]
    dst_dir = os.path.join(args.out_dir, dst_dir)

    calib_h = 128 if 128 // args.out_height > 0 else args.height * 2

    dataset.transform = transforms.Compose([
        font_t.RenderImage(font_path, calib_threshold=0.8, pad=20, calib_h=calib_h),
        font_t.RandomRotation(3),
        font_t.RandomWarping(grid_shape=(5, 2)),
        font_t.GaussianBlur(kernel_size=3),
        font_t.RandomBackground(args.bgs),
        font_t.TailorTensor(pad=3, tailor_x=False, tailor_y=False),
        font_t.ToCustomTensor(),
        # font_t.GrayscaleErosion(kernel_size=2, p=0.05),
        font_t.GrayscaleDilation(kernel_size=2, p=0.1),

        font_t.RandomColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0),
        font_t.Resize(args.out_height),
        font_t.SaveHistory(dst_dir, out_type='img_bk_mask'),
    ])

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                         num_workers=int(args.workers), worker_init_fn=WrkSeeder,
                                         collate_fn=dataset.time_collate_fn)
    go_through(loader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fonts', type=str, default='fonts')
    parser.add_argument('--bgs', type=str, default='backgrounds/paper')
    parser.add_argument('--out_dir', type=str, default='/nas/softechict-nas-3/fquattrini/font_dataset')
    parser.add_argument('--words', type=str, default='words.json')
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--workers', default=0, type=int)
    # parser.add_argument('--seed', default=742, type=int)  # NU
    parser.add_argument('--font_id', type=int)

    parser.add_argument('--num_fonts', type=int, default=300)
    parser.add_argument('--generate_square_dataset', action='store_true')
    parser.add_argument('--num_words_per_font', type=int, default=300)
    parser.add_argument('--out_height', type=int, default=64)
    parser.add_argument('--font_split_id', type=int)
    parser.add_argument('--font_split_size', type=int)

    args = parser.parse_args()
    # args.num_gpu = torch.cuda.device_count()

    assert os.path.isdir(args.fonts)
    assert os.path.isdir(args.bgs)
    os.makedirs(args.out_dir, exist_ok=True)

    # rSeed(args.seed)

    # Shuffle the fonts list
    available_fonts = [
        os.path.join(args.fonts, font) for font in os.listdir(args.fonts) if font.endswith(('.ttf', '.otf'))]

    random.shuffle(available_fonts)
    available_fonts = available_fonts[:args.num_fonts]

    num_words = len(available_fonts) if args.generate_square_dataset else args.num_words_per_font
    dataset = FontsDataset(num_words, None)

    words_path = os.path.join(args.out_dir, args.words)
    if os.path.exists(words_path):
        with open(words_path, 'r') as f:
            print('Loading words from dataset')
            words = json.load(f)  # FIXME PROBLEMA CON MULTITHREAD

        dataset.words = words
        # dataset.alph = sorted(set(''.join(dataset.words)))  # NU
    else:
        with open(words_path, 'w') as f:
            print('Saving words from dataset')
            json.dump(dataset.words, f)

    if args.font_id is not None:
        font_dict = {int(os.path.basename(font).split('_', 1)[0]): font for font in available_fonts}
        available_fonts = [font_dict[args.font_id], ]

    if args.font_split_id is not None:
        font_dict = {int(os.path.basename(font).split('_', 1)[0]): font for font in available_fonts}
        start = args.font_split_id * args.font_split_size
        end = start + args.font_split_size
        available_fonts = [font_dict[font_id] for font_id in range(start, end)]

    for i, font_path in enumerate(available_fonts):
        save_db(dataset, font_path, args)
        print(f'[{i / len(available_fonts) * 100.0:.05f} %] {os.path.basename(font_path)}')
