import io
import time
import json
from typing import Iterator
import torch
import requests
import warnings
import numpy as np
from pathlib import Path
from torchvision.transforms import ToTensor
from torch.utils.data import IterableDataset, DataLoader


class Font2(IterableDataset):
    def __init__(self, path, transform=ToTensor(), nameset='train', fonts_ids=None,
                 words_ids=None, store_on_disk=False, auto_download=False, shuffle=False):
        super(Font2).__init__()
        self.blocks_base_url = 'https://github.com/aimagelab/font_square/releases/download/Dataset/'

        self.path = Path(path)
        self.transform = transform
        self.store_on_disk = store_on_disk
        self.auto_download = auto_download
        self.blocks_path = self.path / 'blocks'
        self.idx = None
        self.shuffle = shuffle
        
        self.num_workers = None
        self.worker_id = None
        self.worker_blocks = None
        self.worker_sizes = None

        with open(self.path / 'fonts.json') as f:
            fonts = json.load(f)
        self.fonts = {int(font): fonts[font] for font in fonts}
        self.fonts_ids = set(fonts_ids) if fonts_ids is not None else set(self.fonts.keys())
    
        with open(self.path / 'words.json') as f:
            words = json.load(f)
        self.words = {int(word): words[word] for word in words}
        self.words_ids = set(words_ids) if words_ids is not None else set(self.words.keys())

        with open(self.path / 'splits.json') as f:
            self.blocks = json.load(f)[nameset]
        self.blocks, self.sizes = zip(*self.blocks)

    
    def load_block(self, idx=0):
        assert self.worker_blocks is not None, 'You must call assign_blocks before load_block'
        block_to_load = self.blocks_path / self.worker_blocks[idx]
        if not block_to_load.exists() and self.auto_download:
            data = self.download(block_to_load)
            data = np.load(io.BytesIO(data.content))
        elif block_to_load.exists():
            data = np.load(block_to_load)
        else:
            raise FileNotFoundError(f'Block {block_to_load} not found. Download it manually from the repo or set auto_download=True.')

        self.images = data['images']
        self.images_wide = data['width']
        self.images_wide = np.cumsum(self.images_wide)
        self.images_wide = np.concatenate([np.array([0,], dtype=self.images_wide.dtype), self.images_wide])
        self.images_idx = data['idx']
    
    def download(self, block):
        url = self.blocks_base_url + block.name
        response = requests.get(url)
        response.raise_for_status()
        block.parent.mkdir(exist_ok=True, parents=True)
        if self.store_on_disk:
            with open(block, "wb") as f:
                f.write(response.content)
        return response
    
    def collate_fn(self, batch):
        imgs, widths, font_ids, words = zip(*batch)
        max_width = max(img.shape[2] for img in imgs)
        imgs = [torch.nn.functional.pad(img, (0, int(max_width - img.shape[2]))) for img in imgs]
        imgs = torch.stack(imgs)
        widths = torch.Tensor(widths)
        font_ids = torch.Tensor(font_ids)
        return imgs, widths, font_ids, words
    
    def __getitem__(self, idx):
        width_start = self.images_wide[idx]
        width_end = self.images_wide[idx + 1]
        width = width_end - width_start
        img = self.images[:, width_start:width_end, :]
        img_idx = self.images_idx[idx]

        font_id = img_idx // len(self.words)
        word = self.words[img_idx % len(self.words)]

        if self.transform is not None:
            img = self.transform(img)

        return img, width, font_id, word
    
    def assign_blocks(self):
        worker_info = torch.utils.data.get_worker_info()
        self.num_workers = 1 if worker_info is None else worker_info.num_workers
        self.worker_id = 0 if worker_info is None else worker_info.id
        self.worker_blocks = list(self.blocks[self.worker_id::self.num_workers])
        self.worker_sizes = list(self.sizes[self.worker_id::self.num_workers])
        
        if self.shuffle:
            blocks_sizes = list(zip(self.worker_blocks, self.worker_sizes))
            np.random.shuffle(blocks_sizes)
            self.worker_blocks, self.worker_sizes = zip(*blocks_sizes)
    
    def __iter__(self):
        self.assign_blocks()
        
        if len(self.worker_blocks) > 0:
            self.idx = 0
            self.load_block(0)
        
        while len(self.worker_blocks) > 0:
            if self.idx >= len(self.images_idx):
                self.worker_blocks.pop(0)
                self.worker_sizes.pop(0)
                self.idx = 0
                
                if len(self.worker_blocks) == 0:
                    break
                self.load_block(0)
                
            img_idx = self.images_idx[self.idx]
            font_id = img_idx // len(self.words)
            word_id = img_idx % len(self.words)
            
            if font_id in self.fonts_ids and word_id in self.words_ids:
                yield self[self.idx]
            self.idx += 1

if __name__ == '__main__':
    db = Font2('.', store_on_disk=True, auto_download=True, nameset='test')
    loader = DataLoader(db, batch_size=32, num_workers=2, collate_fn=db.collate_fn)
    start = time.perf_counter()
    counter = 0 
    for i, (imgs, widths, font_ids, words) in enumerate(loader):
        # print(imgs.shape, widths.shape, font_ids.shape, len(words))
        counter += len(words)
        print(f'\r{counter}', end='')
    print(f'Elapsed time: {time.perf_counter() - start:.2f}s')