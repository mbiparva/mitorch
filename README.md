# miTorch: Medical Imaging in PyTorch
By Mahdi Biparva (PhD in Computer Science)

This package implements deep learning modules for medical imaging application in PyTorch. It contains different modules in the data-pipeline such as the data-loaders, data-containers, transformations etc. In the model-pipeline, there are several segmentation neural networks, training logics, loss function, metrics etc.


### License
TBC

### Citing miTorch
If you find "miTorch: Medical Imaging in PyTorch" useful in your research, please consider citing the research paper:

TBC

<!--    @InProceedings{some_abbreviation,-->
<!--    author = {lname1, fname1 and lname2, fname2},-->
<!--    title = {miTorch: Medical Imaging in PyTorch},-->
<!--    booktitle = {Some venue},-->
<!--    month = {Month},-->
<!--    year = {Year}-->
<!--    }-->

## Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements: Software](#requirements-software)
4. [Requirements: Hardware](#requirements-hardware)
5. [Installation](#installation)
6. [Prerequisites](#prerequisites)
7. [Preparation](#preparation)
8. [Demo: 3D Medical Segmentation](#demo-3d-medical-segmentation)
9. [Demo: Self-Supervised Pre-Training](#demo-self-supervised-pre-training)
10. [Future Work](#future-work)
11. [Contributors](#contributors)

## Introduction
The primary goal is to have a solid, readable, modular, reliable, extendable 
PyTorch package for medical imaging application in deep learning. 
The input is 3D volumes of various modalities and the task could be segmentation, classification, and transfer learning. 

To name a few, the learning tasks are:
  * 3D Medical Segmentation:
     * Head-From-Brain / Skull-Stripping (HFB)
     * White-Matter-Hyperintensities (WMH)
  * Robustness analysis test pipeline
  * Self-supervised learning

## Features

miTorch has currently the following capabilities and components:
* Robust 3D data loading modules for:
    * CT/MRI datasets:
      * Skull-stripping
      * White-matter hyperintensities
    * Electron-Microscopy datasets:
      * Neuron segmentation (counting)
      * Axon/Virus Tracing (Tractography)
      * Hippocampal Subfield Segmentation (multi-label)
* 3D data transformations and pipeline generation:
  * Spatial:
    * Cropping
    * Resizing
    * Axis Rotations
    * Flipping
    * Affine Transformations (Translation, Rotation, Scale, Shear)
  * Intensity:
    * Additive noise (Gaussian, Rice, etc)
    * Corrections (Gamma, Brightness, Contrast)
    * Bias Field
    * Blur
* Automatic Data transformation randomization
* Modular data-pipeline prototyping on-the-fly
* Seamless online patching and batching mechanisms
* Automatic Test Pipeline Generation and Evaluation
* Model Zoo containing:
  * Unet3D
  * Unet3D++ (NestedUnet3D)
  * CBAM
  * DUNet
  * DenseNet
  * SENet
  * VNet
  * DYNUnet
  * HighresNet
* Seamless 3D to 2D network conversion capability
* Various losses such as:
  * Dice
  * Focal
  * Hausdorff
  * Lovasz
  * MSE
* Weighted multi-loss training
* Various metrics such as
  * Jaccard
  * Hausdorff
  * Dice
  * F1
  * Relative Volume Difference
* Hyper-parameter optimization modes:
  * Manual grid search
  * Bayesian semi-automatic optimization search (GPyTorch|BoTorch|Ax)
  * Visualization and Logging (Tensorboard)
* Test Evaluation:
  * Automatic test transformation pipeline generation
  * Batch evaluation and result logging for analysis
* Checkpointing models
* Logging and Visualization (using Tensorboard)
* Automatic-Mixed-Precision (AMP) feature
* Data-Distributed feature (supporting AMP):
* Data Parallel (with GIL)
* Distributed Data Parallel (no GIL, multi node multi GPU)
* Model-Parallel (Under development)


## Requirements: Software
Currently it relies on Python and PyTorch ecosystem.

## Requirements: Hardware
GPU devices with CUDA capabilities are required.

## Installation
There is no installation needed at this moment. You would simply need to call the main function.

### Prerequisites
* Python 3.7
* PyTorch 1.4.0 (not tested on higher versions)
* CUDA 10.0 or higher

## Preparation
TBC

## Demo: 3D Medical Segmentation
TBC

## Demo: Self-Supervised Pre-Training
TBC

## Future Work
We are aiming to develop self-supervised learning modules to enhance the segmentation robustness.

## Contributors
  * Mahdi Biparva (core modeling and development)
  * Parsa Esfahanian (self-supervised development)
  * Braedyn Au (Tracing Segmentation development)
  * Parisa Mojiri (experimentation)
  * Lyndon Boone (experimentation)
  * Maged Goubran (abstraction and methodologies)
