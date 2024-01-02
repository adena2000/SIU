# SIU for Human Pose Estimation

This repository contains the official PyTorch implementation of the following paper:

#### Eliminating Semantic Ambiguity in Human Pose Estimation via Stable Image Upsampling

Shu Jiang, Dong Zhang, Member, IEEE, Rui Yan, Pingcheng Dong, Long Chen, Xiaoyu Du
Computer Science and Engineering, Nanjing University of Science and Technology 

## Abstract 
<p align="justify">

## The overall architecture
![The overall architecture](https:)<br>

## Qualitative results
![Qualitative results](https:)<br>

## Quantitative results and training weights<br>
We provide training weights of SIU with SimpleBaseline, HRNet, and EfficientViT-L-SAM as the baseline.<br>

| Methods | Backbone | Input | Size | body AP (%)| foot AP (%)| face AP (%)| hand AP (%)| whole-body AP (%)| FLOPs | Params | FPS (f/s)|
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |  ---: |

## Installation<br>
#### - Install SIU-main from source<br>
```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision -c pytorch
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
git clone git@github.com:adena2000/SIU-main.git
cd mmpose
pip install -r requirements.txt
pip install -v -e . 
```
#### - Prepare COCO dataset<br>
Prepare [COCO](https://cocodataset.org/#download) in mmpose/data/cocoataset<br>

## Usage
#### - To train the model, please run:
```
bash ./tools/dist_train.sh [--CONFIG_FILE] [--GPU_NUM]
```
#### - To test the model, please run:
```
bash ./tools/dist_test.sh [--CONFIG_FILE] [--GPU_NUM] [--GPU_NUM]
```

## Acknowledgement<br>

## Bibtex
If you find this work is useful for your research, please cite our paper:<br>