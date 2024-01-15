# SIU for Human Pose Estimation

This repository contains the official PyTorch implementation of the following paper:

#### Eliminating Semantic Ambiguity in Human Pose Estimation via Stable Image Upsampling

Shu Jiang, Dong Zhang, Member, IEEE, Rui Yan, Pingcheng Dong, Long Chen, Xiaoyu Du
Computer Science and Engineering, Nanjing University of Science and Technology 

## Abstract 
<p align="justify">

## The overall architecture
![The overall architecture](https://github.com/adena2000/SIU/blob/main/assets/overall.png)<br>

## Qualitative results
![Qualitative results](https://github.com/adena2000/SIU/blob/main/assets/results.png)<br>

## Quantitative results and training weights<br>
We provide training weights of SIU with SimpleBaseline, HRNet, and EfficientViT-L-SAM as the baseline.<br>

| Methods | Backbone | Input Size | body AP (%)| foot AP (%)| face AP (%)| hand AP (%)| whole-body AP (%)| FLOPs | Params | FPS (f/s)| weights |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: |
| SimpleBaseline| ResNet-50 | 256 × 192| 65.2 | 61.5 | 60.6 | 46.0 | 52.1 | 5.5G |34.0M | 1212.6 | - | 
| SIUours | ResNet-50 | 256 × 192 | 66.7 | 62.8 | 61.6 | 48.3 | 53.7 | 4.2G | 23.6M | 1517.0| [weight](https:) | 
| SimpleBaseline | ResNet-50 | 384 × 288 | 66.6 | 63.4 | 73.1 | 53.6 | 57.4 | 12.5G | 34.0M | 558.4 | - | 
| SIUours | ResNet-50 | 384 × 288 | 67.4 | 66.3 | 73.0 | 54.7 | 58.1 | 9.4G | 23.6M | 700.9 | [weight](https:) | 
| SimpleBaseline | ResNet-101 | 256 × 192 | 66.9 | 63.7 | 61.1 | 46.4 | 53.1 | 9.2G | 53.0M | 804.3 | - | 
|SIUours | ResNet-101 |256 × 192 |68.9 | 66.8 | 62.5 | 49.6 | 55.4 | 7.8G |42.5M |926.7|[weight](https:)| 
|SimpleBaseline |ResNet-101 |384 × 288 |69.2 | 68.0 | 74.6 | 54.8 | 59.7 | 20.7G | 53.0M | 370.4 | - | 
|SIUours |ResNet-101 |384 × 288 |70.8 | 70.9 | 74.8 | 56.8 | 61.2 | 17.6G |42.5M |428.6|[weight](https:)| 
|HRNet |HRNet-W32 |256 × 192 |67.8 | 54.3 | 63.0 | 46.7 | 53.6 | 7.7G |28.5M |750.2| - | 
|SIUours |HRNet-W32 |256 × 192 |69.4 | 46.1 | 63.4 | 46.8 | 54.3 | 7.4G |26.6M |753.0|[weight](https:)| 
|HRNet |HRNet-W32 |384 × 288 |70.0 | 58.5 | 72.6 | 51.5 | 58.6 | 17.3G |28.5M |337.3|-| 
|SIUours |HRNet-W32 |384 × 288 |71.4 | 59.6 | 72.2 | 51.3 | 59.4 | 16.6G |26.6M |337.9|[weight](https:)| 
|HRNet |HRNet-W48 |256 × 192 |70.1 |67.5 |65.6 |53.5 |57.9 |15.8G |63.6M |430.0|-| 
|SIUours |HRNet-W48 |256 × 192 |71.0 | 71.1 | 65.1 | 54.4 | 58.6 | 15.1G |59.2M |436.5|[weight](https:)| 
|HRNet |HRNet-W48 |384 × 288| 72.2 | 69.6 | 77.6 | 58.8 | 63.2 | 35.5G |63.6M |196.2|-| 
|SIUours |HRNet-W48 |384 × 288| 73.0 | 74.5 | 77.4 | 58.8 | 64.0 | 33.9G| 59.2M |199.4|[weight](https:)| 
|EfficientViT-L0-SAM |EfficientViT-L0 |384 × 288 |70.5 | 71.1 | 75.3 | 56.4 | 61.2 | 12.7G |32.0M |526.8|[weight](https:)| 
|SIUours |EfficientViT-L0 |384 × 288 |71.0 | 70.9|75.6 |57.8| 61.8 | 10.3G | 27.9M |616.0 |[weight](https:)| 
|EfficientViT-L1-SAM |EfficientViT-L1 |384 × 288 |71.1 | 71.7 | 75.6 | 56.5 | 61.1 | 14.2G |42.3M |460.3|[weight](https:)| 
|SIUours |EfficientViT-L1 |384 × 288 |71.1 | 70.8 | 76.3 | 57.4 | 61.8  | 11.8G| 38.1M | 528.0|[weight](https:)| 

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
Prepare [COCO](https://cocodataset.org/#download) in /path/to/your/data/coco.<br>
COCO-WholeBody annotations for [Train](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/js20_connect_hku_hk/EfE4vxMce2NNiEfJUySLTmwBS5Ay2rbp5-7sHxN6BoldFw?e=tKTLi2) / [Validation](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/js20_connect_hku_hk/EQuxJ51ZSXVPv6EeGnLT65YBvkaVQLAMRYW6pnk6sobfPA?e=jjV2u4)<br>

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
 Thanks [MMPose](https://github.com/open-mmlab/mmpose) teams for the wonderful open source project!
## Bibtex
If you find this work useful for your research, please cite our paper:<br>
