import json
from typing import Iterator

import PIL
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import nltk
import torch
import time
import os
import string
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch.utils.data.sampler import Sampler, T_co
import numpy as np
from cvl import CVLDatasetWords

# NU
# class BigFontsDataset(Dataset):
#     def __init__(self, db_path, transforms=None, nameset='train', fonts_ids=(), words_ids=(), rand_words_count=None):
#         all_words_path = os.path.join(db_path, f'words.json')
#         nameset = '' if nameset is None else '_' + nameset
#         words_path = os.path.join(db_path, f'words{nameset}.json')
#         fonts_path = os.path.join(db_path, f'fonts{nameset}.json')
#         assert os.path.isdir(db_path)
#         assert os.path.exists(words_path)
#         assert os.path.exists(fonts_path)
#
#         with open(all_words_path, 'r') as f:
#             self.alph = sorted(set(''.join(json.load(f))))
#
#         with open(words_path, 'r') as f:
#             self.words = json.load(f)
#
#         with open(fonts_path, 'r') as f:
#             self.fonts = json.load(f)
#         self.fonts = [os.path.join(db_path, f) for f in self.fonts]
#
#         if len(fonts_ids) > 0:
#             fonts_dict = {int(os.path.basename(f)[:5]): f for f in self.fonts}
#             self.fonts = [fonts_dict[idx] for idx in fonts_ids]
#
#         if len(words_ids) > 0:
#             self.words = sorted(self.words)
#             self.words = [self.words[idx] for idx in words_ids]
#
#         self.db_path = db_path
#         self.transforms = transforms
#         self.rand_words_count = rand_words_count
#
#         self.words_to_idx = dict(zip(self.words, range(len(self.words))))
#         self.fonts_to_idx = dict(zip(self.fonts, range(len(self.fonts))))
#
#     def get_words_idx(self, values):
#         return torch.Tensor([self.words_to_idx[v] for v in values]).long()
#
#     def get_fonts_idx(self, values):
#         return torch.Tensor([self.fonts_to_idx[v] for v in values]).long()
#
#     def __len__(self):
#         words_count = self.rand_words_count if not self.rand_words_count is None else len(self.words)
#         return len(self.fonts) * words_count
#
#     def __getitem__(self, sample):
#         if isinstance(sample, (tuple, list)):
#             assert self.rand_words_count is None, 'Parameter rand_words_count can\'t be used with the BatchSampler'
#             word, font = sample
#         elif isinstance(sample, int):
#             font_idx = sample % len(self.fonts)
#             word_idx = sample // len(self.fonts)
#             font = self.fonts[font_idx]
#             word = self.words[word_idx] if self.rand_words_count is None else random.choice(self.words)
#         else:
#             raise NotImplementedError
#         img_path = os.path.join(font, word + '.jpg')
#         try:
#             img = Image.open(img_path)
#             if self.transforms:
#                 img = self.transforms(img)
#             return img, word, font
#         except UnidentifiedImageError:
#             print(f'Error while reading the image {img_path}')
#             raise
#
#     def collate_fn(self, samples):
#         fonts = [s[2] for s in samples]
#         texts = [s[1] for s in samples]
#         imgs = [s[0] for s in samples]
#         out_width = max([img.shape[-1] for img in imgs])
#         imgs = [F.pad(img, pad=(0, out_width - img.shape[-1], 0, 0)) for img in imgs]
#         return torch.stack(imgs).float(), texts, fonts
#
# class IAM(Dataset):
#     def __init__(self, db_path, transforms=None, nameset='train'):
#         # all_words_path = os.path.join(db_path, f'words_data.json')
#         # assert os.path.exists(all_words_path)
#         #
#         # with open(all_words_path, 'r') as f:
#         #     self.words_data = json.load(f)
#
#         self.words_data = {}
#         if nameset in ['train', 'all']:
#             nameset_src = os.path.join(db_path, 'grrnn-IAM-train-list.txt')
#             assert os.path.exists(nameset_src)
#             with open(nameset_src, 'r') as f:
#                 for line in f.readlines():
#                     line_parts = line.strip().split('/')[-1].split('-')
#                     author = int(line_parts[0])
#                     word = ''.join(line_parts[5:]).replace('.png', '')
#                     name = '-'.join(line_parts[1:5])
#                     self.words_data[name] = {'writer_id': author, 'word': word}
#
#         if nameset in ['test', 'all']:
#             nameset_src = os.path.join(db_path, 'grrnn-IAM-test-list.txt')
#             assert os.path.exists(nameset_src)
#             with open(nameset_src, 'r') as f:
#                 for line in f.readlines():
#                     line_parts = line.strip().split('/')[-1].split('-')
#                     author = int(line_parts[0])
#                     word = ''.join(line_parts[5:]).replace('.png', '')
#                     name = '-'.join(line_parts[1:5])
#                     self.words_data[name] = {'writer_id': author, 'word': word}
#
#         self.words = sorted(self.words_data.keys())
#         self.fonts = sorted(set(val['writer_id'] for val in self.words_data.values()))
#
#         self.db_path = db_path
#         self.transforms = transforms
#
#         self.words_to_idx = dict(zip(self.words, range(len(self.words))))
#         self.fonts_to_idx = dict(zip(self.fonts, range(len(self.fonts))))
#
#     def get_words_idx(self, values):
#         return torch.Tensor([self.words_to_idx[v] for v in values]).long()
#
#     def get_fonts_idx(self, values):
#         return torch.Tensor([self.fonts_to_idx[v] for v in values]).long()
#
#     @staticmethod
#     def filename_info(word_path):
#         data = os.path.basename(word_path)[:-4].split('-')
#         author, book, page, line, word = data[:5]
#         text = '-'.join(data[4:])
#         return int(author), book + '-' + page, int(line), int(word), text
#
#     def __len__(self):
#         return len(self.words)
#
#     def __getitem__(self, idx):
#         img_name = self.words[idx]
#         img_data = self.words_data[img_name]
#         a, b, _, _ = img_name.split('-')
#
#         img_path = os.path.join(self.db_path, 'words', a, f'{a}-{b}', img_name + '.png')
#         assert os.path.exists(img_path)
#         img = Image.open(img_path)
#         if self.transforms:
#             img = self.transforms(img)
#         return img, img_data['word'], img_data['writer_id']
#
#     def collate_fn(self, samples):
#         fonts = [s[2] for s in samples]
#         texts = [s[1] for s in samples]
#         imgs = [s[0] for s in samples]
#         out_width = max([img.shape[-1] for img in imgs])
#         imgs = [F.pad(img, pad=(0, out_width - img.shape[-1], 0, 0)) for img in imgs]
#         return torch.stack(imgs).float(), texts, fonts
#
# class IAMDatasetWords(IAM):
#     def __getitem__(self, idx):
#         img_name = self.words[idx]
#         img_data = self.words_data[img_name]
#         a, b, _, _ = img_name.split('-')
#
#         img_path = os.path.join(self.db_path, 'words', a, f'{a}-{b}', img_name + '.png')
#         assert os.path.exists(img_path)
#         img = Image.open(img_path)
#         if self.transforms:
#             img = self.transforms(img)
#         return img, img_data['word'], img_data['writer_id'], f'{a}-{b}'
#
#     # def __getitem__(self, idx):
#     #     if idx in self.samples: return self.samples[idx]
#     #
#     #     word_path = self.words_path[idx]
#     #     author, page, line, word, text = self.filename_info(word_path)
#     #     img = Image.open(word_path)
#     #
#     #     if self.transforms is not None:
#     #         img = self.transforms(img)
#     #
#     #     return author, page, line, word, text, img, word_path
#
#     def collate_fn(self, samples):
#         imgs = [s[0] for s in samples]
#         authors = [s[2] for s in samples]
#         pages = [s[3] for s in samples]
#         words = [s[1] for s in samples]
#         out_width = max([img.shape[-1] for img in imgs])
#         imgs = [F.pad(img, pad=(0, out_width - img.shape[-1], 0, 0)) for img in imgs]
#         return {'imgs': torch.stack(imgs).float(), 'authors': authors, 'pages': pages, 'words': words}
#
# class CVL(CVLDatasetWords):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.fonts = sorted(set(self.filename_info(info)[0] for info in self.words_path))
#         self.fonts_to_idx = dict(zip(self.fonts, range(len(self.fonts))))
#
#     def get_fonts_idx(self, values):
#         return torch.Tensor([self.fonts_to_idx[v] for v in values]).long()
#
#     def collate_fn(self, samples):
#         authors = [s[0] for s in samples]
#         texts = [s[4] for s in samples]
#         imgs = [s[5] for s in samples]
#         out_width = max([img.shape[-1] for img in imgs])
#         imgs = [F.pad(img, pad=(0, out_width - img.shape[-1], 0, 0)) for img in imgs]
#         return torch.stack(imgs).float(), texts, authors
#
# class CVLgrrnn(Dataset):
#     def __init__(self, db_path, transforms=None, nameset='train'):
#         self.words_data = {}
#         self.img_to_path = {}
#         for nset in ['trainset', 'testset']:
#             nameset_path = os.path.join(db_path, nset, 'words')
#             for auth in os.listdir(nameset_path):
#                 auth_path = os.path.join(nameset_path, auth)
#                 for img in os.listdir(auth_path):
#                     self.img_to_path[img] = os.path.join(auth_path, img)
#
#         if nameset in ['train', 'all']:
#             nameset_src = os.path.join(db_path, 'CVL-train-list.txt')
#             assert os.path.exists(nameset_src)
#             with open(nameset_src, 'r') as f:
#                 for line in f.readlines():
#                     line_parts = line.strip().split('/')[-1].split('-')
#                     author = int(line_parts[0])
#                     word = ''.join(line_parts[4:]).replace('.tif', '')
#                     name = '-'.join(line_parts)
#                     self.words_data[name] = {'writer_id': author, 'word': word}
#
#         if nameset in ['test', 'all']:
#             nameset_src = os.path.join(db_path, 'CVL-test-list.txt')
#             assert os.path.exists(nameset_src)
#             with open(nameset_src, 'r') as f:
#                 for line in f.readlines():
#                     line_parts = line.strip().split('/')[-1].split('-')
#                     author = int(line_parts[0])
#                     word = ''.join(line_parts[4:]).replace('.tif', '')
#                     name = '-'.join(line_parts)
#                     self.words_data[name] = {'writer_id': author, 'word': word}
#
#         self.words = sorted(self.words_data.keys())
#         self.fonts = sorted(set(val['writer_id'] for val in self.words_data.values()))
#
#     def __len__(self):
#         return len(self.words)
#
#     def __getitem__(self, idx):
#         img_name = self.words[idx]
#         img_data = self.words_data[img_name]
#         a, b, _, _ = img_name.split('-')
#
#         img_path = os.path.join(self.db_path, 'words', a, f'{a}-{b}', img_name + '.png')
#         assert os.path.exists(img_path)
#         img = Image.open(img_path)
#         if self.transforms:
#             img = self.transforms(img)
#         return img, img_data['word'], img_data['writer_id']
#
#     def collate_fn(self, samples):
#         fonts = [s[2] for s in samples]
#         texts = [s[1] for s in samples]
#         imgs = [s[0] for s in samples]
#         out_width = max([img.shape[-1] for img in imgs])
#         imgs = [F.pad(img, pad=(0, out_width - img.shape[-1], 0, 0)) for img in imgs]
#         return torch.stack(imgs).float(), texts, fonts

class FontsDataset(Dataset):  # Perchè fonts dataset se contiene solo parole e trasformate da farci sopra?
    def __init__(self, size, transform=None):
        self.transform = transform
        self.words = random.sample(nltk.corpus.words.words(), size)
        # self.alph = sorted(set(''.join(self.words)))  # TODO len = 53, più caratteri  # NU

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        start_time = time.time()
        sample = self.transform(word)
        sample['time'] = time.time() - start_time
        return sample

    # def collate_fn(self, samples):
    #     texts = [s['text'] for s in samples]
    #     imgs = [s['img'] for s in samples]
    #     out_width = max([img.shape[-1] for img in imgs])
    #     imgs = [F.pad(img, pad=(0, out_width - img.shape[-1], 0, 0)) for img in imgs]  # TODO perchè non fai padding di height?
    #     return texts, torch.stack(imgs).float()
    #
    # def dict_collate_fn(self, samples):
    #     return {s['text']: s.to_dict() for s in samples}

    def time_collate_fn(self, samples):
        out_times = []
        for sample in samples:
            times = {key: val['time'] for key, val in sample.history}
            times['total'] = sample['time']
            out_times.append(times)
        return out_times

    # def calibrate(self):
    #     mean, std = self.calculate_mean_std()
    #     self.transform.transforms.append(
    #         T.Normalize(mean=mean, std=std)
    #     )

    # def calculate_mean_std(self):
    #     loader = DataLoader(self, batch_size=4, shuffle=False, num_workers=2, collate_fn=self.collate_fn)
    #
    #     time_start = time.time()
    #     mean = 0.
    #     std = 0.
    #     nb_samples = 0.
    #     for i, sample in enumerate(loader, 1):
    #         _, data = sample
    #         batch_samples = data.size(0)
    #         data = data.view(batch_samples, data.size(1), -1)
    #         mean += data.mean(2).sum(0)
    #         std += data.std(2).sum(0)
    #         nb_samples += batch_samples
    #         print(f'  [{i:04d}/{len(loader)}] mean={mean / nb_samples} std={std / nb_samples} time: {time.time() - time_start:.02f} s'.ljust(os.get_terminal_size().columns), end='\r')
    #     mean /= nb_samples
    #     std /= nb_samples
    #     print(f'  [{i:04d}/{len(loader)}] mean={mean} std={std} time: {time.time() - time_start:.02f} s'.ljust(os.get_terminal_size().columns))
    #     return mean, std


# class FeatureDataset(Dataset):  # NU
#     def __init__(self, db_path, nameset=None):
#         assert os.path.exists(db_path)
#         data = torch.load(db_path)
#         self.preds = data['preds'].detach().cpu()
#         self.words = data['words'].detach().cpu()
#         self.fonts = data['fonts'].detach().cpu()
#
#         self.fonts_classes = self.fonts.unique()
#         self.words_classes = self.words.unique()
#
#         torch.manual_seed(1204)
#         size = len(self.preds)
#         perm = torch.randperm(size)
#         if nameset is not None:
#             size_train = int(size * 0.7)
#             size_test = (size - size_train) // 2
#             size_val = size - size_train - size_test
#             assert size_train + size_test + size_val == size
#             if nameset == 'train':
#                 perm = perm[0: size_train]
#             elif nameset == 'test':
#                 perm = perm[size_test: size_train + size_test]
#             elif nameset == 'val':
#                 perm = perm[-size_val:]
#             else:
#                 raise ValueError
#         self.preds = self.preds[perm]
#         self.words = self.words[perm]
#         self.fonts = self.fonts[perm]
#
#     def __len__(self):
#         return self.preds.shape[0]
#
#     @property
#     def sample_shape(self):
#         return self.preds[0].shape
#
#     def __getitem__(self, index):
#         return self.preds[index], self.words[index], self.fonts[index]
#
# class SquareBatchSampler(Sampler): # NU
#     def __init__(self, data_source, batch_size, shuffle=True, true_shuffle=True, size=None):
#         super().__init__(data_source)
#         assert isinstance(data_source, BigFontsDataset)
#
#         if isinstance(size, int): size = (size, size)
#         if size is None: size = (None, None)
#
#         self.words = data_source.words
#         self.fonts = data_source.fonts
#         self.dataset = data_source
#
#         if true_shuffle:
#             random.shuffle(self.words)
#             random.shuffle(self.fonts)
#
#         if len(self.words) % batch_size != 0:
#             count = len(self.words) % batch_size
#             print(f'WARNING with batch_size {batch_size} you are leaving out {count} words: {self.words[-count:]}')
#             self.words = self.words[:-count]
#         if len(self.fonts) % batch_size != 0:
#             count = len(self.fonts) % batch_size
#             print(f'WARNING with batch_size {batch_size} you are leaving out {count} fonts: {self.fonts[-count:]}')
#             self.fonts = self.fonts[:-count]
#
#         start_time = time.time()
#         self.batch_size = batch_size
#         self.batch_idx = np.arange(len(data_source), dtype=np.uint32)
#         self.batch_idx = self.batch_idx.reshape(len(data_source.words), len(data_source.fonts))
#         self.batch_idx = self.batch_idx[:size[0]:self.batch_size, :size[1]:self.batch_size]
#         # self.batch_idx *= self.size
#         self.batch_idx = self.batch_idx.flatten()
#         if shuffle: np.random.shuffle(self.batch_idx)
#         print(f'Time to generate the batch indices: {time.time() - start_time:.03f} s ',
#               f'mem: {self.batch_idx.size * self.batch_idx.itemsize / 1024 / 1024:.02f} MB')
#
#     def __len__(self):
#         return len(self.batch_idx)
#
#     def __iter__(self):
#         for idx in self.batch_idx:
#             font_id = idx % len(self.fonts)
#             word_id = idx // len(self.fonts)
#             for font in self.fonts[font_id:font_id + self.batch_size]:
#                 for word in self.words[word_id:word_id + self.batch_size]:
#                     yield word, font
#
#     @property
#     def words_used(self):
#         indices = self.batch_idx // len(self.fonts)
#         indices = np.unique(indices)
#
#         # for word in self.words[word_id:word_id + self.size]:
#         words = []
#         for idx in indices:
#             words.extend(self.words[idx:idx + self.batch_size])
#         return sorted(set(words))
#
#     @property
#     def fonts_used(self):
#         indices = self.batch_idx % len(self.fonts)
#         indices = np.unique(indices)
#
#         fonts = []
#         for idx in indices:
#             fonts.extend(self.fonts[idx:idx + self.batch_size])
#         return sorted(set(fonts))

#  # NU
# if __name__ == '__main__':
#     dataset = IAM(db_path='/nas/softechict-nas-2/datasets/iam_words', nameset='test')
#
#     for i in range(len(dataset)):
#         try:
#             sample = dataset[i]
#         except PIL.UnidentifiedImageError:
#             print(f'\nERROR with {dataset.words[i]}')
#         except KeyError:
#             print(f'\nKEY-ERROR with {dataset.words[i]}')
#
#         if i % 1000 == 0: print(f'[{i}/{len(dataset)}] {i / len(dataset) * 100.0:.02f} %')
