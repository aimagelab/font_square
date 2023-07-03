# $Font^2$

This repo contains the code to generate and download the $Font^2$ dataset presented in the paper [Evaluating Synthetic Pre-Training for Handwriting Processing Tasks](https://arxiv.org/abs/2304.01842)

If you find it useful, please cite it as:
```
@article{pippi2023evaluating,
  title={Evaluating Synthetic Pre-Training for Handwriting Processing Tasks},
  author={Pippi, Vittorio and Cascianelli, Silvia and Baraldi, Lorenzo and Cucchiara, Rita},
  journal={Pattern Recognition Letters},
  year={2023},
  publisher={Elsevier}
}
```
To obtain the dataset, please run:
```python
dataset = Font2('path/to/dataset', store_on_disk=True, auto_download=True)
loader = DataLoader(dataset, batch_size=32, num_workers=1, collate_fn=dataset.collate_fn, shuffle=False)
```
This way, the loader will automatically download (`auto_download=True`) and save (`store_on_disk=True`) the data inside the `blocks` folder, at the same level as the `fonts.json` and `words.json` files.
We suggest you to run the above lines by keeping `shuffle=False`: this way, the download will be much quicker.

Please note that you can also download the dataset (already shuffled and with the same augmentations as presented in the paper) manually from the [releases](https://github.com/aimagelab/font_square/releases), where can also find the checkpoints of a ResNet-18 and a VGG-19 trained on it.

## Load checkpoints
We uploaded the checkpoints of the model trained on $Font^2$ in the [releases](https://github.com/aimagelab/font_square/releases)
### ResNet-18
```python
import torch
from torchvision import models

model = models.resnet18(num_classes=10400)
checkpoint = torch.hub.load_state_dict_from_url('https://github.com/aimagelab/font_square/releases/download/ResNet-18/RN18_class_10400.pth')
model.load_state_dict(checkpoint)
```
### VGG-16
```python
import torch
from torchvision import models

model = models.vgg16(num_classes=10400)
checkpoint = torch.hub.load_state_dict_from_url('https://github.com/aimagelab/font_square/releases/download/VGG-16/VGG16_class_10400.pth')
model.load_state_dict(checkpoint)
```
### Inception-v3
```python
import torch
from torchvision import models

model = models.inception_v3(num_classes=10400, init_weights=False, aux_logits=False)
checkpoint = torch.hub.load_state_dict_from_url('https://github.com/aimagelab/font_square/releases/download/Inception-v3/IV3_class_10400.pth')
model.load_state_dict(checkpoint, strict=False)
```
**WARNING**: Since the input size of the `Inception-v3` is $299 \times 299$ we implemented the following transform to adapt the $Font^2$ samples to the network input:
```python
class ToInceptionV3Input:
    def __init__(self, size=299):
        self.size = size

    def __call__(self, x):
        h_rep = math.ceil(self.size / x.shape[1])
        w_rep = math.ceil(self.size / x.shape[2])
        return x.repeat(1, h_rep, w_rep)[:, :self.size, :self.size]
```
The `ToInceptionV3Input` transform repeats the input image vertically and horizontally enough times to cover the whole input image and trim the excess to fit the desired shape.

