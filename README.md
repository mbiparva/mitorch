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
2. [Requirements: Software](#requirements-software)
3. [Requirements: Hardware](#requirements-hardware)
4. [Installation](#installation)
5. [Prerequisites](#prerequisites)
6. [Preparation](#preparation)
7. [Demo: HFB](#demo-hfb-3d-segmentation)
8. [Demo: WMH](#demo-wmh-3d-segmentation)
9. [Future Work](#future-work)
10. [Contributors](#contributors)

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

## Demo: 3D Segmentation
TBC

## Demo: WMH 3D Segmentation
TBC

## Future Work
We are aiming to develop self-supervised learning modules to enhance the segmentation robustness.

## Contributors
  * Mahdi Biparva (core modeling and development)
  * Parsa Esfahanian (self-supervised development)
  * Braedyn Au (Tracing Segmentation development)
  * Parisa Mojiri (experimentation)
  * Maged Goubran (abstraction and methodologies)
