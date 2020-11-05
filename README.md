# Cascaded Partial Decoder for Fast and Accurate Salient Object Detection (CVPR2019)

https://drive.google.com/file/d/1gBFWGrjt0hBIW1I9IOoUJYTF-C1CwPfr/view?usp=sharing

# Documentation
CPD ranks first in the challenging [SOC benchmark](http://dpfan.net/SOCBenchmark/) [2019.11.6].

## Requirements:

Python 3.8.2

PyTorch 1.5.0

TorchVision 0.6.0

## Usage

### Datsets

Datasets to be set in the following directory format:

````
.
├── datasets
    ├── test
    │   ├── DUTS-TE
    │   │   ├── gts
    │   │   │   ├── ILSVRC2012_test_00000003.jpg
    │   │   │   ...
    │   │   │   └── sun_ekmqudbbrseiyiht.jpg
    │   │   └── imgs
    │   │       ├── ILSVRC2012_test_00000003.png
    │   │       ...
    │   │       └── sun_ekmqudbbrseiyiht.png
    │   ├── PASCAL-S
    │       ├── gts
    │       │   ├── 1.jpg
    │       │   ...
    │       │   └── 850.jpg
    │       └── imgs
    │           ├── 1.png
    │           ...
    │           └── 850.png
    └── train
        └── DUTS-TR
            ├── gts
            │   ├── ILSVRC2012_test_00000004.jpg
            │   ...
            │   └── sun_dzkggnowaqnfrorl.jpg
            └── img
                ├── ILSVRC2012_test_00000004.jpg
                │   ...
                └── sun_dzkggnowaqnfrorl.jpg
````
### Training

Run `train.py` to train a new model. All models a saved to `./models/<model_name>`

```
usage: train.py [-h] [--datasets_path DATASETS_PATH] [--device {cuda,cpu}]
                [--attention] [--imgres IMGRES] [--epoch EPOCH] [--lr LR]
                [--batch_size BATCH_SIZE] [--clip CLIP]
                [--decay_rate DECAY_RATE] [--decay_epoch DECAY_EPOCH]

optional arguments:
  -h, --help            show this help message and exit
  --datasets_path DATASETS_PATH
                        path to datasets, default = ./datasets/train
  --device {cuda,cpu}   use cuda or cpu, default = cuda
  --attention           use attention branch model
  --imgres IMGRES       image input and output resolution, default = 352
  --epoch EPOCH         number of epochs, default = 100
  --lr LR               learning rate, default = 0.0001
  --batch_size BATCH_SIZE
                        training batch size, default = 10
  --clip CLIP           gradient clipping margin, default = 0.5
  --decay_rate DECAY_RATE
                        decay rate of learning rate, default = 0.1
  --decay_epoch DECAY_EPOCH
                        every n epochs decay learning rate, default = 50

```

### Testing

Pre-trained model should be placed in the root directory. Run `test.py` to produce ground truths.

```
usage: test.py [-h] [--datasets_path DATASETS_PATH] [--save_path SAVE_PATH]
               [--pth PTH] [--attention] [--imgres IMGRES]

optional arguments:
  -h, --help            show this help message and exit
  --datasets_path DATASETS_PATH
                        path to datasets, default = ./datasets/test
  --save_path SAVE_PATH
                        path to save results, default = ./results
  --pth PTH             model filename, default = CPD.pth
  --attention           use attention branch model
  --imgres IMGRES       image input and output resolution, default = 352
```

## Models

As cited in the paper the following model architectures are as follows.

|Model|Backbone|Attention Branch Only|
|:----|:----|:----|
|CPA|VGG16||
|CPA-A|VGG16|X|
|CPD-R|ResNet50|
|CPD-RA|ResNet50|X|

## Pre-trained model

VGG16     backbone: [google drive](https://drive.google.com/open?id=1ddopz30_sNPOb0MvTCoNwZwL-oQDMGIW), [BaiduYun](https://pan.baidu.com/s/18qF_tpyRfbgZ0YLleP8c5A) (code: gb5u)

ResNet50  backbone: [google drive](https://drive.google.com/open?id=188sybU9VU5rW2BH2Yzhko4w-G5sPp6yG), [BaiduYun](https://pan.baidu.com/s/1tc6MWlj5sbMJJGCyUNFxbQ) (code: klfd)

## Pre-computed saliency maps

VGG16     backbone: [google drive](https://drive.google.com/open?id=1LcCTcKGEsZjO8WUgbGpiiZ4atQrK1u_O)

ResNet50  backbone: [google drive](https://drive.google.com/open?id=16pLY2qYZ1KIzPRwR7zFUseEDJiwhdHOg)

## Performance

Maximum F-measure

|Model|FPS|ECSSD|HKU-IS|DUT-OMRON|DUTS-TEST|PASCAL-S|
|:----|:---:|:---:|:---:|:---:|:---:|:---:|
|PiCANet|7|0.931|0.921|0.794|0.851|0.862|
|CPD|66|0.936|0.924|0.794|0.864|0.866|
|CPD-A|105|0.928|0.918|0.781|0.854|0.859|
|PiCANet-R|5|0.935|0.919|0.803|0.860|0.863|
|CPD-R|62|0.939|0.925|0.797|0.865|0.864|
|CPD-RA|104|0.934|0.918|0.783|0.852|0.855|

MAE

|Model|ECSSD|HKU-IS|DUT-OMRON|DUTS-TEST|PASCAL-S|
|:----|:---:|:---:|:---:|:---:|:---:|
|PiCANet|0.046|0.042|0.068|0.054|0.076|
|CPD|0.040|0.033|0.057|0.043|0.074|
|CPD-A|0.045|0.037|0.061|0.047|0.077|
|PiCANet-R|0.046|0.043|0.065|0.051|0.075|
|CPD-R|0.037|0.034|0.056|0.043|0.072|
|CPD-RA|0.043|0.038|0.059|0.048|0.077|

## Shadow Detection

pre-computed maps: [google drive](https://drive.google.com/open?id=1R__w0FXpMhUMnIuoxPaX6cFzwAypX13U)

## Performance

BER

|Model|SBU|ISTD|UCF|
|:----|:----|:----|:----|
|DSC|5.59|8.24|8.10|
|CPD|4.19|6.76|7.21|

# Citation
```
@InProceedings{Wu_2019_CVPR,
author = {Wu, Zhe and Su, Li and Huang, Qingming},
title = {Cascaded Partial Decoder for Fast and Accurate Salient Object Detection},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

# Changelog

## 1.2 - 2019-05-31
### Added
- model/CPD_model.py - VGG16 model with only the attention branch

### Changed
- train.py - Now able to train attention only model
- train.py - Organised parse argument names to be more clear

### Fixed
- gts_folder.py - Error with os.walk


## 1.1 - 2019-05-30
### Added
- gts_folder.py - New torchvision dataset. Loads images from multiple datasets
- train.py - Dataset root parser argument
- train.py - CUDA parser argument

### Changed
- train.py - train_loader now uses new trochvison dataset


## 1.0 - 2019-05-26
### Added
- vgg.py - PyTorch Hub to fetch pre-trained VGG-16 model
- test_CPD.py - Use CUDA only if available
- test_CPD.py - Progress print out

### Changed
- test_CPD.py - Corrected dataset paths

### Fixed
- test_CPD.py - Replaced deprecated torch.nn.functional.upsample with torch.nn.functional.interpolate
- test_CPD.py - Replaced deprecated scipy.misc.imsave with torchvision.utils.save_image
- CPD_models.py - Relative import in Python 3
- CPD_ResNet_models.py - Relative import in Python 3
