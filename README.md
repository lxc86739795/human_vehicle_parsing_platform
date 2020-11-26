# parsing_platform
A pytorch codebase for human parsing and vehicle parsing.

## Introduction
A pytorch codebase for human parsing and vehicle parsing.

## Requirements

- Linux or macOS with python ≥ 3.6
- PyTorch = 0.4.1
- torchvision that matches the Pytorch installation. You can install them together at [pytorch.org](https://pytorch.org/) to make sure of this.
- tensorboard (needed for visualization): `pip install tensorboard`

## Supported methods

- [x] PSPNet
- [x] DeepLabV3
- [x] CCNet
- [x] DANet
- [x] OCNet
- [x] CE2P
- [x] HRNet
- [x] BraidNet


## Supported datasets

- [x] LIP
- [x] MVP

## Citation
```BibTeX

@inproceedings{mm/LiuZLSM19,
  author    = {Xinchen Liu and
               Meng Zhang and
               Wu Liu and
               Jingkuan Song and
               Tao Mei},
  title     = {BraidNet: Braiding Semantics and Details for Accurate Human Parsing},
  booktitle = ACM MM,
  pages     = {338--346},
  year      = {2019}
}

@inproceedings{mm/LiuLZY020,
  author    = {Xinchen Liu and
               Wu Liu and
               Jinkai Zheng and
               Chenggang Yan and
               Tao Mei},
  title     = {Beyond the Parts: Learning Multi-view Cross-part Correlation for Vehicle
               Re-identification},
  booktitle = {ACM MM},
  pages     = {907--915},
  year      = {2020}
}
