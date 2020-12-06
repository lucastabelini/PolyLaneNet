<div align="center">

# PolyLaneNet
![Method overview](figures/method-overview.png "Method overview")
</div>

## Description
Code for the [PolyLaneNet paper](https://arxiv.org/abs/2004.10924 "PolyLaneNet paper"), accepted to ICPR 2020, by [Lucas Tabelini](https://github.com/lucastabelini), [Thiago M. Paixão](https://sites.google.com/view/thiagopx), [Rodrigo F. Berriel](http://rodrigoberriel.com), [Claudine Badue](https://www.inf.ufes.br/~claudine/),
[Alberto F. De Souza](https://inf.ufes.br/~alberto), and [Thiago Oliveira-Santos](https://www.inf.ufes.br/~todsantos/home).

**News**: The source code for our new state-of-the-art lane detection method, LaneATT, has been released. Check it out [here](https://github.com/lucastabelini/LaneATT/).

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Reproducing the paper results](#reproducing)

<a name="installation"/>

### Installation
The code requires Python 3, and has been tested on Python 3.5.2, but should work on newer versions of Python too.

Install dependencies:
```
pip install -r requirements.txt
```

<a name="usage"/>

### Usage
#### Training
Every setting for a training is set through a YAML configuration file.
Thus, in order to train a model you will have to setup the configuration file.
An example is shown:
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

With the config file created, run the training script:
```bash
python train.py --exp_name tusimple --cfg config.yaml
```
This script's options are:
```
  --exp_name            Experiment name.
  --cfg                 Config file for the training (.yaml)
  --resume              Resume training. If a training session was interrupted, run it again with the same arguments and this option to resume the training from the last checkpoint.
  --validate            Wheter to validate during the training session. Was not in our experiments, which means it has not been thoroughly tested.
  --deterministic       set cudnn.deterministic = True and cudnn.benchmark = False
```

#### Testing
After training, run the `test.py` script to get the metrics:
```bash
python test.py --exp_name tusimple --cfg config.yaml --epoch 2695
```
This script's options are:
```
  --exp_name            Experiment name.
  --cfg                 Config file for the test (.yaml). (probably the same one used in the training)
  --epoch EPOCH         Epoch to test the model on
  --batch_size          Number of images per batch
  --view                Show predictions. Will draw the predictions in an image and then show it (cv.imshow)
```

If you have any issues with either training or testing feel free to open an issue.

<a name="reproducing"/>

### Reproducing the paper results

#### Models
All models trained for the paper can be found [here](https://drive.google.com/open?id=1oyZncVnUB1GRJl5L4oXz50RkcNFM_FFC "Models on Google Drive").

#### Datasets
- [TuSimple](https://github.com/TuSimple/tusimple-benchmark "TuSimple")
- [ELAS](https://github.com/rodrigoberriel/ego-lane-analysis-system/tree/master/datasets "ELAS")
- [LLAMAS](https://unsupervised-llamas.com/llamas/ "LLAMAS")

#### How to
To reproduce the results, you can either retrain a model with the same settings (which should yield results pretty close to the reported ones) or just test the model.
If you want to retrain, you only need the appropriate YAML settings file, which you can find in the `cfgs` directory.
If you just want to reproduce the exact reported metrics by testing the model, you'll have to:
1. Download the experiment directory. You don't need to download all model checkpoints if you want, you'll only need the last one (`model_2695.pt`, with the exception of the experiments on ELAS and LLAMAS).
1. Modify all path related fields (i.e., dataset paths and `exps_dir`) in the `config.yaml` file inside the experiment directory.
1. Move the downloaded experiment to your `exps_dir` folder.

Then, run:

```bash
python test.py --exp_name $exp_name --cfg $exps_dir/$exp_name/config.yaml --epoch 2695
```
Replacing `$exp_name` with the name of the directory you downloaded (the name of the experiment) and `$exps_dir` with the `exps_dir` value you defined inside the `config.yaml` file. The script will look for a directory named `$exps_dir/$exp_name/models` to load the model.


