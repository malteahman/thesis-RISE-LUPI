# mleo_masters
The Power of Privilege: Enhancing Land Cover Classification with Privileged Information by Agnes Eriksson and Malte Åhman. Here follows a file description. 

# Config folder - IMPORTANT FOLDER
Folder with all config files, these will be commented in further detail in the end.

# Dataset folder

## Paths folder
Paths contain all paths for the used 5 % of the data, to assure the same images was used on every 5 % run. Will have to be regenerated if file structure is different.

## example_set.py
Outdated for inspiration.

## read_dataset.py - IMPORTANT FILE
File called from loop2.py to read all data. Reads in paths, loads in images in __get_item__, applies transformations, normalizes etc. 

# fcnpytorch folder
Folder with FCN8-s model (copied from github). Called from support_functions_loop > set_model

# Scripts folder

## Ensamble models
Scripts to run the various ensemble models. _1 is the first attempt, _2 is a cleaned version for 5 %, _3 is using only three models for 5%, _100 is the same as _2 but for 100 % models.

## flair_play.py
Example from Aleksis.

## loop.py
Old loop file.

## process_tensorboard.py
File to read certain values from tensorboard, and do histograms etc.

## script_to_extract_instances.py
Script to extract instances for copy_and_paste method.

## script_to_save_paths.py
Script to save paths for any subset size of the data. Used to generate the paths in the dataset folder.

## stats.py
Used to calculate mean, std, for all channels. 

# Transform folder

## basic.py
File to apply augmentations (used in read_dataset.py).

# Main folder

## custom_losses.py
Our custom made loss functions.

## eval_model.py
File used for evaluation of models, called from main_test.py, when setting config.eval.eval_type as not a test type. I.e. runs on the validation set.

## loop2.py - IMPORTANT FILE
Main file for running tranings, called from main.py. 

## main_test.py - IMPORTANT FILE
File to call to run tests or evaluations. Configs are set in the normal config file (sorry about that).

## main.py - IMPORTANT FILE
File to call to run any tranings. Configs are set in config file.

## support_functions_logging.py
Support functions for loggins various metrics and images. Called from loop2.py.


## support_functions_loop.py - IMPORTANT FILE
Support functions for setting net, loss function and calculating losses during training. Called from loop2.py.


## support_functions_noise.py 
Support functions for the various noise architectures. Called from loop2.py.

## test.py 
File used for test of models, called from main_test.py, when setting config.eval.eval_type as a test type. I.e. runs on the test set.

## unet_module.py
All custom made U-Nets. 

## util.py
Support.

# Config > config.yaml - IMPORTANT

## Hydra settings
hydra:
  run:
    dir: dir where everything should be saved
defaults:
  - _self_
  - dataset: which yaml file to use for dataset configs
  - transform: which yaml file to use for tranformations

## General model settings
- seed: seed (42)
- device: cuda:#
- use_transform: True or False, for using transforms or not
- batch_size: batch size (8 has been used)
- eval_every: how often to run throguh validation set, only integers are valid
- save_model_freq: how often to save model, unless it is the best model, then it is always saved
- val_batch_size: validation batch size (2 has been used)
- test_batch_size: test batch size (2 has been used)
- num_workers: for reading in data
- lr: learning rate (0.0002 has been used)
- max_epochs: max number of epochs to train (200-300 commonly for 5% runs, a bit less for 100%)
- save_optimizer: Set this to true if you want to be able to keep training the model at a later stage, as of now the code has no support for this.
- weight_decay: weight decay (0 used in general 1e-4 used for networks which are not U-Net based)
- optimizer: optimizwe, support for adam and SGD (adam has been used)
- beta1: parameter for optimzer (0.9 has been used)
- momentum: parameter for adam (0.9 has been used)
- loss_function: the loss function to use, for a list of available loss functions, see support_functions_loop > set_loss()

## Special stuff

- indices_to_use: which indices to use, when only looking att some indices, only compatible with loss_function CE_ignoring_classes
- ignore_last_classes: If to ignore the last 7 classes, only implemented for CE_tversky, and is bad for that as well

- label_smoothing: smoothing label, only for CE_tversky, CE (only half implemented for CE_tversky)
- copy_and_paste: probability for an image to get a copy and paste crop (0.5 or 0.0 has been used), TURN OF JITTERING BEFORE USAGE
- frac_copy_samples: fraction of the available crops to use, as it should not be too many according to paper (0.4 has been used) 

- loss_w: for loss combos between CE, Dice and/or focal
  - ce_w: weight for CE
  - focal_w: weight for Focal
  - The rest is Dice, regardless of the combo

## Model specific stuff

- model:
  - n_class: number of classes to use, 13 for grouped, 19 for all (19 is most commonly used)
  - n_channels: number of channels to include in importad images (from 1-5)
  - channels: which specific channels (any subset of [0,1,2,3,4] is fine, should align with n_channels)
  - name: which model to use, for a list of available models, see support_functions_loop > set_model()
  - use_pretrained_net: if the network we should train is a pretrained net, so that we have to read it in (True or False)

  - overwrite: if model is of owerwrite type (i.e. load privileged weights to unpriveleged model)
    - overwriting_name: which specific overwrite
    - overwriting_net: path to specific net to overwrite new net with 

  - pretrained: if the model to train should be pretrained
   - pretrained_name: what is the model type of the net to read in
   - training_path: path to specific pretrained net

  - unet_predict_priv: if model is of predict priv type (either normal or reversed)
    - unet_channels: number of channels to use as input

  - teacher_student: if model is of teacher student type
    - student_name: student network type (not all are supported, but most)
    - teacher_name: teacher network type (not all are supported, but most)
    - teacher_path: path to specific teacher
    - teacher_channels: number channels the teacher will have access to 
    - teacher_spec_channels: which specific channels
    - student_channels: number channels the student will have acces to
    - student_spec_channels: which specific channels
    - alpha: parameter for weighting between the two loss componenets (commonly 0.4)
    - ts_loss: loss function between teacher prediction and student prediction (commonly KL)
    - student_loss: loss function between student prediction and ground truth (tversky for U-Net, CE otherwise isch)
    - student_T: student temperature
    - teacher_T: teacher temperature
    - R: number to divide ts_loss component with, to match scale of student_loss 

  - multi_teacher: if model is of multi teacher type, currently maximum of two teachers possible, see teacher_student for explanation of settings, teacher is expected to be unet
    - student_name: 
    - teacher_1_path: 
    - teacher_2_path: 
    - teacher_1_channels: 
    - teacher_1_spec_channels: 
    - teacher_2_channels: 
    - teacher_2_spec_channels:
    - student_channels: 
    - student_spec_channels: 
    - alpha: 
    - ts_loss: 
    - student_T: 
    - teacher_T: 
    - R: 


  - resnet50: special for resnet50
    - pretrained: DeepLabV3_ResNet50_Weights.DEFAULT #None does not work, but they seem to want it
    - pretrained_backbone: ResNet50_Weights.IMAGENET1K_V2 #default, can also use enum, read docs
 
  - mtd: if any of the metadata network
    - reweight_late: if we should reweight the prediction based on classes, if using late fusion (commonly False)
    - mtd_weighting: weight for metadata prediction in late fusion (0.3 commonly)
    - linear_mtd_preprocess: if preprocessing metadata through linear net (True or False)
    - feature_block: if mid fusion used, which feature block to insert it into, using SE blocks

- noise: if we want noise!
  - noise: True or False, if we want noise or not
  - noise_type: which noise type (see support_functions_noise for availibility)
  - noise_distribution_type: batch or image
  - stepwise_linear_function: which speed to introduce noise (see support_functions_noise for availability)

## Evaluation stuff

- eval: if calling from main_test.py (assure model.n_channels and model.channels are what you want them to be)
  - eval_type: for possible choices, see test.py and eval_model.py, primarely test for test, and normal for a validation set run
  - eval_model: network type for model to evaluate
  - eval_loss: loss function to log and report
  - eval_channels: num input channels to model
  - eval_path: path to model to test/eval

# Config > dataset > read_dataset.yaml
- script_location: path to script to read in data (in this case 'dataset/read_dataset.py')
- path: path to where the dataset is located
- norm_path: path to normalizing stats (mean, std), not relevant in this case
- X_path: path to inputs, relative to "path"
- Y_path: path to labels, relative to "path"
- path_to_metadata: path to metadata, relative to "path"
- det_crop: for deterministic crops, should be False
- crop_step_size: step size for if we crop deterministically
- crop_size: size of crops, if we crop, for both det_crop and random_crop (commonly not used)
- random_crop: if we want to crop (False is used)
- scale: if we want to scale to sixe which is a power of 2 (only relevant if cropping, should be False)
- dataset_size: fraction of dataset size to use, only adapted for 0.05 or 1.0, since we otherwise can not guarantee the same fraction each time
- senti_path: path to sentinel images, relative to "path"
- senti_size: size of satellite images, onesided from middle (16 has been used)
- label_mask: if we want to include labels as input during training (commonly False)

- mean: [113.7693, 118.0902, 109.2753, 102.3808, 16.7343] #B is the first channel, then G, then R (only on train, not val)
- std: [52.3925, 46.0153, 45.2657, 39.4576, 29.5914]

- mtd_mean: [662249.4503979404, 6597921.059597216, 232.49435711211387, 229.81857652905612, 658.4401564490374]
- mtd_std: [197046.23959320912, 245498.4174863277, 383.5279207331584, 42.59626831271271, 130.18012431546725]

- mean_senti: [126.5045, 127.2261, 127.4363, 127.2859, 127.0794, 127.1531, 127.1539, 127.0784, 127.0547, 127.1054]
- std_senti: [44.7545, 44.3896, 44.5939, 43.8597, 43.7466, 43.6991, 43.6935, 43.6990, 43.7319, 43.8538]

# Config > transform > basic.yaml
Should be self explanatory

# Config > TS_RGB_config.yaml 
Configs for running teacher student with RGB teacher and RGB student. Set dataset size in read_dataset.yaml. Model paths available in https://docs.google.com/spreadsheets/d/1KGY8hdrcdzMBhgpEluJOk_uNwQak59cH32duaFDaz9Q/edit#gid=1976657697. 

# Config > TS_LUPI_config.yaml 
Configs for running teacher student with priv teacher and RGB student (LUPI). Set dataset size in read_dataset.yaml. Model paths available in https://docs.google.com/spreadsheets/d/1KGY8hdrcdzMBhgpEluJOk_uNwQak59cH32duaFDaz9Q/edit#gid=1976657697. 

# Config > TS_priv_config.yaml 
Configs for running teacher student with priv teacher and priv student. Set dataset size in read_dataset.yaml. Model paths available in https://docs.google.com/spreadsheets/d/1KGY8hdrcdzMBhgpEluJOk_uNwQak59cH32duaFDaz9Q/edit#gid=1976657697. 

# Paths to models
Access this file for paths to models https://docs.google.com/spreadsheets/d/1KGY8hdrcdzMBhgpEluJOk_uNwQak59cH32duaFDaz9Q/edit#gid=1976657697. In the path column is a date, and in the "Dator" column it is specified if the model is available on aleksis or dgx1. For full paths
- dgx1: /raid/dlgroupmsc/logs/2024-02-09_10-11-25 (change datetime to change model)
- aleksis: /Projects/master-theses/agnes-malte-spring-2024/log_res/2024-02-08_17-09-25 (change datetime to change model)


# About hydra
https://hydra.cc/docs/intro/
