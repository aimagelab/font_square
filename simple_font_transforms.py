import pickle
import time
from typing import Sequence
from .render_font import Render
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import os
from PIL import Image
import json

# class ToTensor(T.ToTensor):  # NU
#     def __call__(self, sample):
#         img, txt = sample
#         img = super(ToTensor).__call__(img)
#         return img, txt