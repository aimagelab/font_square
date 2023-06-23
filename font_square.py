import io
import time
import json
import torch
import requests
import warnings
import numpy as np
from pathlib import Path
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader


class Font2(Dataset):
    def __init__(self, path, transform=ToTensor(), store_on_disk=False, auto_download=False):
        super().__init__()

        self.path = Path(path)
        self.transform = transform
        self.store_on_disk = store_on_disk
        self.auto_download = auto_download
        self.blocks_path = self.path / 'blocks'
        self.start_idx = 0

        with open(self.path / 'fonts.json') as f:
            fonts = json.load(f)
        self.fonts = {int(font): fonts[font] for font in fonts}
    
        with open(self.path / 'words.json') as f:
            words = json.load(f)
        self.words = {int(word): words[word] for word in words}

        self.block_size = 200_000
        self.blocks_base_url = 'https://github.com/aimagelab/font_square/releases/download/Dataset/'
        self.blocks = []
        for x in range(0, len(self.words) * len(self.fonts), self.block_size):
            start = x
            end = min(x + self.block_size, len(self.words) * len(self.fonts))
            self.blocks.append(f'{start:09}_{end:09d}.npz')
        self.loaded_block = None
        self.last_loaded_time = None
        self.load_next()


    def __len__(self):
        return len(self.words) * len(self.fonts) - self.start_idx
    
    def load_next(self, idx=0):
        if self.loaded_block == idx // self.block_size:
            return
        if self.last_loaded_time is not None and time.perf_counter() - self.last_loaded_time < 10:
            warnings.warn('Loading blocks too fast. You are not reading the dataset '
                          'sequentially, this method will heavly slow down the performance.')
        self.last_loaded_time = time.perf_counter()

        curr_block = self.blocks_path / self.blocks[idx // self.block_size]
        if not curr_block.exists() and self.auto_download:
            data = self.download(curr_block)
            data = np.load(io.BytesIO(data.content))
        elif curr_block.exists():
            data = np.load(curr_block)
        else:
            raise FileNotFoundError(f'Block {curr_block} not found. Download it manually from the repo or set auto_download=True.')

        self.images = data['images']
        self.images_wide = data['width']
        self.images_wide = np.cumsum(self.images_wide)
        self.images_wide = np.concatenate([np.array([0,], dtype=self.images_wide.dtype), self.images_wide])
        self.images_idx = data['idx']
        self.loaded_block = idx // self.block_size
    
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
        idx = idx + self.start_idx
        self.load_next(idx)

        idx = idx % self.block_size
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

if __name__ == '__main__':
    db = Font2('.', store_on_disk=True, auto_download=True)
    loader = DataLoader(db, batch_size=32, num_workers=1, collate_fn=db.collate_fn, shuffle=False)
    start = time.perf_counter()
    for i, (imgs, widths, font_ids, words) in enumerate(loader):
        print(imgs.shape, widths.shape, font_ids.shape, len(words))
        if i > 1000:
            break
    print(f'Elapsed time: {time.perf_counter() - start:.2f}s')