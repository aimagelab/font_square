import pickle
import time
from typing import Sequence
from pathlib import Path
from render_font import Render
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import os
from PIL import Image
from tps import TPS
import json


def mask_coords(mask):
    x0 = mask.max(0).values.type(torch.uint8).argmax()
    y0 = mask.max(1).values.type(torch.uint8).argmax()
    x1 = mask.shape[1] - mask.max(0).values.flip(0).type(torch.uint8).argmax()
    y1 = mask.shape[0] - mask.max(1).values.flip(0).type(torch.uint8).argmax()
    return x0, y0, x1, y1


class Sample:
    def __init__(self, **kwargs):
        self.data = kwargs
        self.history = []

    def __dict__(self):
        return self.data

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __str__(self):
        tmp_data = self.data.copy()
        tmp_data['history'] = self.history
        return str(tmp_data)

    def record(self, transform, **kwargs):
        self.history.append((transform.__class__.__name__, kwargs))

    def to_dict(self):
        return self.history

    @staticmethod
    def from_dict(data):
        raise NotImplementedError


class RenderImage(object):
    def __init__(self, font_path, height=None, width=None, calib_text=None, calib_threshold=0.7, calib_h=128, pad=0):
        self.width = width
        self.height = height
        self.pad = pad
        self.calib_text = calib_text
        self.calib_threshold = calib_threshold
        self.calib_h = calib_h
        self.font_data_path = 'fonts_data.json'

        def __render(_path, font_size=64, calibrate=True):
            return Render(_path, height, width, font_size, calibrate, calib_text, calib_threshold, calib_h)

        self.__render = __render

        if isinstance(font_path, (list, tuple)):
            self.renders = [__render(path) for path in font_path]
        elif isinstance(font_path, str) and os.path.isfile(font_path):
            self.renders = [__render(font_path), ]
        elif isinstance(font_path, str) and os.path.isdir(font_path):
            self.renders = self.load(font_path)
            self.save(font_path)
        else:
            raise NotImplementedError

    def load(self, src_dir):
        if os.path.exists(os.path.join(src_dir, self.font_data_path)):
            with open(os.path.join(src_dir, self.font_data_path), 'r') as f:
                data = json.load(f)
            res = []
            full_paths = [os.path.join(src_dir, path) for path in os.listdir(src_dir) if not path.endswith('.json')]
            for path in full_paths:
                font_size, calibrate = 64, True
                if path in data: font_size, calibrate = data[path], False
                res.append(self.__render(path, font_size, calibrate))
            return res
        else:
            return [self.__render(os.path.join(src_dir, path)) for path in os.listdir(src_dir) if
                    not path.endswith('.json')]

    def save(self, src_dir):
        with open(os.path.join(src_dir, self.font_data_path), 'w') as f:
            json.dump({render.font_path: render.font_size for render in self.renders}, f)

    def __call__(self, text):
        start_time = time.time()
        font_render = random.sample(self.renders, 1)[0]
        np_img = font_render.render(text, return_np=True, action='center_left', pad=self.pad)
        sample = Sample(text=text, font_img=torch.from_numpy(np_img))
        sample.record(self, font_path=font_render.font_path, font_size=font_render.font_size,
                      width=self.width, height=self.height, pad=self.pad, calib_text=self.calib_text,
                      calib_threshold=self.calib_threshold, calib_h=self.calib_h, time=time.time() - start_time)
        return sample

    @staticmethod
    def deterministic(data):
        pass


class RandomResizedCrop:
    def __init__(self, ratio_eps, scale):
        self.scale = scale
        self.ratio_eps = ratio_eps

    def __call__(self, sample):
        img = sample['font_img']
        ratio = img.shape[1] / img.shape[0]
        ratio = (ratio - self.ratio_eps, ratio + self.ratio_eps)

        i, j, h, w = T.RandomResizedCrop.get_params(img, self.scale, ratio)
        sample['font_img'] = F.resized_crop(img.unsqueeze(0), i, j, h, w, img.shape).squeeze(0)
        sample.record(self, ijhw=(i, j, h, w))
        return sample


class RandomWarping:
    def __init__(self, std=0.05, grid_shape=(5, 3)):
        self.std = std
        self.grid_shape = grid_shape

    def __call__(self, sample):
        start_time = time.time()
        h, w = sample['font_img'].shape
        x = np.linspace(-1, 1, self.grid_shape[0])
        y = np.linspace(-1, 1, self.grid_shape[1])
        xx, yy = np.meshgrid(x, y)

        # make source surface, get uniformed distributed control points
        source_xy = np.stack([xx, yy], axis=2).reshape(-1, 2)

        # make deformed surface
        deform_xy = source_xy + np.random.normal(scale=self.std, size=source_xy.shape)
        # deform_xy = np.stack([xx, yy], axis=2).reshape(-1, 2)

        # get coefficient, use class
        trans = TPS(source_xy, deform_xy)

        # make other points a left-bottom to upper-right line on source surface
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)
        test_xy = np.stack([xx, yy], axis=2).reshape(-1, 2)

        # get transformed points
        transformed_xy = trans(test_xy)
        grid = torch.from_numpy(transformed_xy)
        grid = grid.reshape(1, h, w, 2)
        img = sample['font_img'].unsqueeze(0).unsqueeze(0).type(grid.dtype)
        sample['font_img'] = torch.nn.functional.grid_sample(img, grid, mode='nearest',
                                                             padding_mode='border', align_corners=False).squeeze()
        sample.record(self, deform_xy=deform_xy.tolist(), time=time.time() - start_time)
        return sample


class ToRGB:
    def __call__(self, img):
        rgb_img = Image.new("RGB", img.size)
        rgb_img.paste(img)
        return rgb_img


class ImgResize:
    def __init__(self, height):
        self.height = height

    def __call__(self, img):
        _, h, w = img.shape
        if h == self.height: return img
        out_w = int(self.height * w / h)
        return F.resize(img, [self.height, out_w])


class Resize(ImgResize):
    def __call__(self, sample):
        start_time = time.time()
        sample['font_img'] = super().__call__(sample['font_img'].unsqueeze(0)).squeeze(0)
        sample['bg'] = super().__call__(sample['bg'])
        sample['img'] = super().__call__(sample['img'])
        sample.record(self, height=self.height, time=time.time() - start_time)
        return sample


# class RandomVerticalResizedCrop:
#     def __init__(self, scale):
#         assert isinstance(scale, Sequence)
#         self.scale = scale
#
#     @staticmethod
#     def get_params(img, scale):
#         img_h, img_w = img.shape
#         trg_scale = torch.empty(1).uniform_(scale[0], scale[1]).item()
#         trg_h = int(img_h * trg_scale)
#         trg_i = torch.empty(1).uniform_(0, img_h - trg_h).item()
#         return trg_i, 0, trg_h, img_w
#
#     def __call__(self, sample):
#         raise NotImplementedError
#         text, img = sample
#         i, j, h, w = self.get_params(img, self.scale)
#         return text, F.resized_crop(img, i, j, h, w, self.size, self.interpolation)


# class RandomHorizontalResizedCrop:  # NU
#     def __init__(self, scale):
#         assert isinstance(scale, Sequence)
#         self.scale = scale
#
#     def __call__(self, sample):
#         raise NotImplementedError
#         text, img = sample
#         i, j, h, w = RandomVerticalResizedCrop.get_params(img, self.scale)
#         return text, F.resized_crop(img, i, j, h, w, self.size, self.interpolation)


class RandomBackground(object):
    start_time = time.time()

    def __init__(self, bg_path, include_white=True):
        if isinstance(bg_path, (list, tuple)):
            self.bgs_path = [path for path in bg_path]
        elif isinstance(bg_path, str) and os.path.isfile(bg_path):
            self.bgs_path = [bg_path, ]
        elif isinstance(bg_path, str) and os.path.isdir(bg_path):
            self.bgs_path = [os.path.join(bg_path, path) for path in os.listdir(
                bg_path)]  # from pathlib import Path; [str(path) for path in Path(gb_path).rglob('*.png')]
        else:
            raise NotImplementedError
        self.bgs = [Image.open(path) for path in self.bgs_path]
        self.bgs = [F.to_tensor(img.convert('RGB')) for img in self.bgs]
        self.include_white = include_white
        if include_white:
            max_height = max([bg.shape[1] for bg in self.bgs])
            max_width = max([bg.shape[2] for bg in self.bgs])
            self.bgs.append(torch.ones((3, max_height, max_width), dtype=torch.float32))
            self.bgs_path.append('white')

    @staticmethod
    def random_patch(bg, img_h, img_w):
        _, bg_h, bg_w = bg.shape
        resize_crop = T.RandomResizedCrop((img_h, img_w))
        i, j, h, w = resize_crop.get_params(bg, resize_crop.scale, resize_crop.ratio)
        return F.resized_crop(bg, i, j, h, w, resize_crop.size, resize_crop.interpolation), (i, j, h, w)

    def __call__(self, sample):
        start_time = time.time()
        font_img = sample['font_img']
        assert len(font_img.shape) == 2
        img_h, img_w = font_img.shape

        bg_idx = random.randrange(0, len(self.bgs))
        patch, ijhw = self.random_patch(self.bgs[bg_idx], img_h, img_w)

        flip_ud, flip_lf = False, False
        if random.random() > 0.5:
            patch, flip_ud = patch.flip(1), True  # up-down
        if random.random() > 0.5:
            patch, flip_lf = patch.flip(2), True  # left-right

        sample['bg'] = patch
        sample.record(self, background=self.bgs_path[bg_idx], ijhw=ijhw, flip_ud=flip_ud, flip_lf=flip_lf,
                      time=time.time() - start_time)
        return sample


class GaussianBlur(T.GaussianBlur):
    def forward(self, sample):
        start_time = time.time()
        sample['font_img'] = super().forward(sample['font_img'].unsqueeze(0)).squeeze()
        sample.record(self, kernel_size=self.kernel_size, time=time.time() - start_time)
        return sample


class TailorTensor:
    def __init__(self, pad=0, tailor_x=True, tailor_y=True):
        self.pad = pad
        self.tailor_x = tailor_x
        self.tailor_y = tailor_y

    def __call__(self, sample):
        start_time = time.time()
        font_img, bg = sample['font_img'], sample['bg']
        x0, y0, x1, y1 = mask_coords(font_img < 0.99)

        x0 = max(0, x0 - self.pad)
        x1 = min(bg.shape[2], x1 + self.pad)
        y0 = max(0, y0 - self.pad)
        y1 = min(bg.shape[1], y1 + self.pad)

        if not self.tailor_x:
            x0, x1 = 0, bg.shape[2]
        if not self.tailor_y:
            y0, y1 = 0, bg.shape[1]

        sample['font_img'] = font_img[y0:y1, x0:x1]
        sample['bg'] = bg[:, y0:y1, x0:x1]
        sample.record(self, pad=self.pad, time=time.time() - start_time)
        return sample


class ToCustomTensor:
    def __init__(self, min_alpha=0.5, max_alpha=1.0):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def __call__(self, sample):
        start_time = time.time()
        alpha = random.uniform(self.min_alpha, self.max_alpha)
        font_mask = 1 - ((1 - sample['font_img']) * alpha)
        sample['img'] = font_mask.unsqueeze(0) * sample['bg']
        sample.record(self, time=time.time() - start_time)
        return sample


# class Normalize(T.Normalize):  # NU
#     def forward(self, sample):
#         text, img = sample
#         img = super().forward(img)
#         return text, img


class RandomRotation(T.RandomRotation):
    def __init__(self, degrees, *args, **kwargs):
        super().__init__(degrees, *args, **kwargs)

    def forward(self, sample):
        start_time = time.time()
        font_img = sample['font_img']
        angle = self.get_params(self.degrees)

        # sample['font_img'] = F.rotate(1 - font_img.unsqueeze(0), angle, self.resample, self.expand, self.center, self.fill)
        # TODO DONE resample does not work anymore

        # Rotate adds black background in the new areas
        sample['font_img'] = F.rotate(1 - font_img.unsqueeze(0), angle, self.interpolation, self.expand, self.center,
                                      self.fill)

        sample['font_img'] = 1 - sample['font_img'].squeeze(0)
        sample.record(self, angle=angle, time=time.time() - start_time)
        return sample


class RandomColorJitter(T.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness, contrast, saturation, hue)

    def forward(self, sample):
        start_time = time.time()
        brightness_factor = None
        contrast_factor = None
        saturation_factor = None
        hue_factor = None
        img = sample['img']

        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)

        if brightness_factor is not None or contrast_factor is not None or saturation_factor is not None or hue_factor is not None:
            sample['img'] = img
            sample.record(self, fn_idx=fn_idx.tolist(), brightness_factor=brightness_factor,
                          contrast_factor=contrast_factor,
                          saturation_factor=saturation_factor, hue_factor=hue_factor, time=time.time() - start_time)
        return sample


# class RandomAdjustSharpness(T.RandomAdjustSharpness):
#     def forward(self, sample):
#         text, img = sample
#         return text, super().forward(img)


# class RandomGrayscale(T.RandomGrayscale):  # NU
#     def forward(self, sample):
#         sample['img'] = super().forward(sample['img'])
#         return sample


# class RandomSolarize(T.RandomSolarize):
#     def forward(self, sample):
#         text, img = sample
#         return text, super().forward(img)


# class RandomInvert(T.RandomSolarize):
#     def __init__(self, p=0.5):
#         super().__init__(0, p)
#
#     def forward(self, sample):
#         text, img = sample
#         return text, super().forward(img)


# class RandomAffine(T.RandomAffine):  # NU
#     def forward(self, sample):
#         text, img = sample
#         channels = img.shape[0]
#         if channels == 3:
#             img = super().forward(img)
#         elif channels == 4:
#             img[0] = super().forward(img[0].unsqueeze(0))
#         else:
#             raise NotImplementedError
#         return text, img

# class GrayscaleErosion:  # NU
#     def __init__(self, kernel_size=5, p=0.5):
#         self.kernel_size = kernel_size
#         self.p = p
#
#     def __call__(self, sample):
#         if random.random() > self.p:
#             start_time = time.time()
#             pad = self.kernel_size // 2
#             img = torch.nn.functional.pad(sample['img'], (pad, pad, pad, pad), mode='constant', value=0.0)
#             img = torch.nn.functional.max_pool2d(img, self.kernel_size, stride=1)
#             sample['img'] = img
#             sample.record(self, kernel_size=self.kernel_size, time=time.time() - start_time)
#         return sample

class GrayscaleDilation:
    def __init__(self, kernel_size=5, p=0.5):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            start_time = time.time()
            # pad = self.kernel_size // 2  # The output size was different than input size

            pad = (self.kernel_size - 1)
            img = torch.nn.functional.pad(-sample['img'], ((pad - 1), 1, (pad - 1), 1), mode='constant', value=-1.0)
            img = -torch.nn.functional.max_pool2d(img, self.kernel_size, stride=1)
            sample['img'] = img
            sample.record(self, kernel_size=self.kernel_size, time=time.time() - start_time)
        return sample


class SaveHistory:
    def __init__(self, out_dir, out_type):
        self.out_dir = out_dir
        self.out_type = out_type
        os.makedirs(out_dir, exist_ok=True)

        if self.out_type == 'img_bk_mask':
            self.mask_dir = Path(out_dir, 'mask')
            self.bg_dir = Path(out_dir, 'bg')
            self.full_dir = Path(out_dir, 'full')
            os.makedirs(self.mask_dir, exist_ok=True)
            os.makedirs(self.bg_dir, exist_ok=True)
            os.makedirs(self.full_dir, exist_ok=True)

    def __call__(self, sample):
        path = Path(self.out_dir, sample['text'])
        if self.out_type == 'json':
            with open(path + '.json', 'w') as f:
                json.dump(sample.to_dict(), f)
        elif self.out_type == 'pickle':
            with open(os.path.join(self.out_dir, sample['text']) + '.pkl', 'wb') as f:
                pickle.dump(sample.to_dict(), f)
        elif self.out_type == 'png':
            F.to_pil_image(sample['img']).save(path + '.png')
        elif self.out_type == 'jpg':
            F.to_pil_image(sample['img']).save(path + '.jpg')
        elif self.out_type == 'img_bk_mask':
            F.to_pil_image(sample['img']).save(Path(self.full_dir, f"{sample['text']}.jpg"))
            F.to_pil_image(sample['font_img']).save(Path(self.mask_dir, f"{sample['text']}_mask.png"))
            F.to_pil_image(sample['bg']).save(Path(self.bg_dir, f"{sample['text']}_bg.jpg"))
        else:
            raise NotImplementedError
        return sample
