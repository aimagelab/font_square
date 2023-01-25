import json

from torch.utils.data import Dataset
import random
import torch
import os
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
from ds_fonts.font_transforms import ImgResize
import time

class CEDAR(Dataset):
    def __init__(self, db_path, transforms=None, nameset='org', load_in_mem=False):
        self.words_path = []

        if nameset in ['org', 'all']:
            src_dir = os.path.join(db_path, 'full_org')
            self.words_path += [os.path.join(src_dir, word) for word in os.listdir(src_dir) if word.endswith('.png')]

        if nameset in ['forg', 'all']:
            src_dir = os.path.join(db_path, 'full_forg')
            self.words_path += [os.path.join(src_dir, word) for word in os.listdir(src_dir) if word.endswith('.png')]

        self.db_path = db_path
        self.transforms = transforms

        self.load_in_mem = False
        self.samples = {}
        if load_in_mem:
            print(f'Loading in mem {len(self)} samples')
            for idx in range(len(self)):
                self.samples[idx] = self[idx]
                if idx % (len(self) // 10) == 0: print(f'Sample [{idx}]')
        self.load_in_mem = load_in_mem

    @staticmethod
    def filename_info(word_path):
        _, author, sign_id = os.path.basename(word_path)[:-4].split('_')
        return int(author), int(sign_id)

    def __len__(self):
        return len(self.words_path)

    def __getitem__(self, idx):
        if idx in self.samples: return self.samples[idx]

        word_path = self.words_path[idx]
        author, sign_id = self.filename_info(word_path)
        img = Image.open(word_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, author, sign_id

    def collate_fn(self, samples):
        imgs = [s[0] for s in samples]
        authors = [s[1] for s in samples]
        sign_ids = [s[2] for s in samples]
        out_width = max([img.shape[-1] for img in imgs])
        imgs = [F.pad(img, pad=(0, out_width - img.shape[-1], 0, 0)) for img in imgs]
        return {'imgs': torch.stack(imgs).float(), 'authors': authors, 'sign_ids': sign_ids}


class CEDAR_font2(CEDAR):
    def __init__(self, db_path, transforms=None, nameset='org', authors_ids=(), sign_ids=()):
        super(CEDAR_font2, self).__init__(db_path, transforms, nameset, False)

        if len(authors_ids) > 0:
            self.words_path = [word for word in self.words_path if self.filename_info(word)[0] in authors_ids]

        if len(sign_ids) > 0:
            self.words_path = [word for word in self.words_path if self.filename_info(word)[1] in sign_ids]

        self.fonts = sorted(set(self.filename_info(word)[0] for word in self.words_path))

    def get_fonts_idx(self, fonts_idx):
        return torch.tensor(fonts_idx) - 1

    def collate_fn(self, samples):
        imgs = [s[0] for s in samples]
        authors = [s[1] for s in samples]
        sign_ids = [s[2] for s in samples]
        out_width = max([img.shape[-1] for img in imgs])
        imgs = [F.pad(img, pad=(0, out_width - img.shape[-1], 0, 0)) for img in imgs]
        return torch.stack(imgs).float(), sign_ids, authors


if __name__ == '__main__':
    t = T.Compose([
        T.ToTensor(),
        ImgResize(32)
    ])

    s_time = time.time()
    cedar = CEDAR_font2(
        'D:\\CEDAR',
        transforms=t,
        nameset='org',
        authors_ids=(),
        sign_ids=(1, 2, 3, 4, 5)
    )
    print(f'done in {time.time() - s_time}')
