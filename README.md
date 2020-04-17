# PolyLaneNet
Code for the under-review PolyLaneNet paper. This page is a work-in-progress.

## Table of Contents
1. [Reproducing the paper results](#reproducing)
2. [Installation](#installation)
3. [Usage](#usage)

<a name="reproducing"/>

### Reproducing the paper results
#### Models
Models trained for the paper will be available soon.
#### Datasets
- [TuSimple](https://github.com/TuSimple/tusimple-benchmark "TuSimple")
- [ELAS](https://github.com/rodrigoberriel/ego-lane-analysis-system/tree/master/datasets "ELAS")
- [LLAMAS](https://unsupervised-llamas.com/llamas/ "LLAMAS")

<a name="installation"/>

### Installation
Install dependencies:
```
pip install -r requirements.txt
```

That's all. A Docker container and a Google Colab notebook will be available soon.

<a name="usage"/>

### Usage
#### Training
First, create the experiment configuration file. An example is shown:
```yaml
# Training settings
exps_dir: 'experiments' # Path to the root for the experiments directory (not only the one you will run)
iter_log_interval: 1 # Log training iteration every N iterations
iter_time_window: 100 # Moving average iterations window for the printed loss metric
model_save_interval: 1 # Save model every N epochs
seed: 0 # Seed for randomness
backup: drive:polylanenet-experiments # The experiment directory will be automatically uploaded using rclone after the training ends. Leave empty if you do not want this.
model:
  name: PolyRegression
  parameters:
    num_outputs: 35 # (5 lanes) * (1 conf + 2 (upper & lower) + 4 poly coeffs)
    pretrained: true
    backbone: 'efficientnet-b0'
    pred_category: false
loss_parameters:
  conf_weight: 1
  lower_weight: 1
  upper_weight: 1
  cls_weight: 0
  poly_weight: 300
batch_size: 16
epochs: 2695
optimizer:
  name: Adam
  parameters:
    lr: 3.0e-4
lr_scheduler:
  name: CosineAnnealingLR
  parameters:
    T_max: 385

# Testing settings
test_parameters:
  conf_threshold: 0.5 # Set predictions with confidence lower than this to 0 (i.e., set as invalid for the metrics)

# Dataset settings
datasets:
  train:
    type: PointsDataset
    parameters:
      dataset: tusimple
      split: train
      img_size: [360, 640]
      normalize: true
      aug_chance: 0.9090909090909091 # 10/11
      augmentations: # ImgAug augmentations
       - name: Affine
         parameters:
           rotate: !!python/tuple [-10, 10]
       - name: HorizontalFlip
         parameters:
           p: 0.5
       - name: CropToFixedSize
         parameters:
           width: 1152
           height: 648
      root: "datasets/tusimple" # Dataset root

  test: &test
    type: PointsDataset
    parameters:
      dataset: tusimple
      split: val
      img_size: [360, 640]
      root: "datasets/tusimple"
      normalize: true
      augmentations: []

  # val = test
  val:
    <<: *test
```

With the config file created, start training:
```bash
python train.py --exp_name tusimple --cfg config.yaml
```
#### Testing
```bash
python test.py --exp_name tusimple --cfg config.yaml --epoch 2695
```
