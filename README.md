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

