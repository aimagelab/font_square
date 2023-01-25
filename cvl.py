import json

from torch.utils.data import Dataset
import random
import torch
import os
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
from font_transforms import ImgResize
import time

class CVLDatasetWords(Dataset):
    def __init__(self, db_path, transforms=None, nameset='train', pages=(), authors=(), load_in_mem=False):
        assert nameset in ['train', 'test', 'all']
        trainset_path = os.path.join(db_path, 'trainset', 'words')
        testset_path = os.path.join(db_path, 'testset', 'words')
        self.words_path = []
        if nameset in ['train', 'all']:
            self.words_path += [os.path.join(trainset_path, auth, word) for auth in os.listdir(trainset_path) for word in os.listdir(os.path.join(trainset_path, auth))]
        if nameset in ['test', 'all']:
            self.words_path += [os.path.join(testset_path, auth, word) for auth in os.listdir(testset_path) for word in os.listdir(os.path.join(testset_path, auth))]
        # self.words_path = sum(self.words_path, [])

        self.tree = {}
        for word_path in self.words_path:
            assert word_path.endswith('.tif')
            author, page, line, word, text = self.filename_info(word_path)

            if author not in authors and len(authors) > 0: continue
            if page not in pages and len(pages) > 0: continue

            if not author in self.tree:
                self.tree[author] = {}
            if not page in self.tree[author]:
                self.tree[author][page] = []
            self.tree[author][page].append(word_path)

        self.words_path = sorted(set(sum([page for pages in self.tree.values() for page in pages.values()], [])))
        self.db_path = db_path
        self.transforms = transforms

        self.load_in_mem = False
        self.samples = {}
        if load_in_mem:
            print(f'Loading in mem {len(self)} samples')
            for idx in range(len(self)):
                self.samples[idx] = self[idx]
                print(f'Sample [{idx}]', end='\r')
        self.load_in_mem = load_in_mem

    @staticmethod
    def filename_info(word_path):
        data = os.path.basename(word_path)[:-4].split('-')
        author, page, line, word = data[:4]
        text = '-'.join(data[4:])
        return int(author), int(page), int(line), int(word), text

    def __len__(self):
        return len(self.words_path)

    def __getitem__(self, idx):
        if idx in self.samples: return self.samples[idx]

        word_path = self.words_path[idx]
        author, page, line, word, text = self.filename_info(word_path)
        img = Image.open(word_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return author, page, line, word, text, img, word_path

    def collate_fn(self, samples):
        authors = [s[0] for s in samples]
        pages = [s[1] for s in samples]
        lines = [s[2] for s in samples]
        words = [s[3] for s in samples]
        texts = [s[4] for s in samples]
        paths = [s[6] for s in samples]
        imgs = [s[5] for s in samples]
        out_width = max([img.shape[-1] for img in imgs])
        imgs = [F.pad(img, pad=(0, out_width - img.shape[-1], 0, 0)) for img in imgs]
        return {'imgs': torch.stack(imgs).float(), 'authors': authors, 'pages': pages,
                'lines': lines, 'words': words, 'texts': texts, 'paths': paths}

if __name__ == '__main__':
    t = T.Compose([
        T.ToTensor(),
        ImgResize(32)
    ])

    s_time = time.time()
    cvl = CVLDatasetWords(
        '/mnt/beegfs/work/FoMo_AIISDH/scascianelli/CVL/cvl-database-1-1',
        transforms=t,
        nameset='all',
        pages=(1, ),
        load_in_mem=True
    )
    print(f'done in {time.time() - s_time}')