
<p align="center">

  <h2 align="center"><strong>Scaling Efficient Masked Autoencoder Learning on Large Remote Sensing Dataset</strong></h2>

  <p align="center">
     Fengxiang Wang<sup>1</sup>&nbsp;&nbsp;&nbsp;
    Hongzhen Wang<sup>2,‡</sup>&nbsp;&nbsp;&nbsp;
    Di Wang<sup>3</sup>&nbsp;&nbsp;&nbsp;
    Zonghao Guo<sup>4</sup></br>
    Zhenyu Zhong<sup>5</sup>&nbsp;&nbsp;&nbsp;
    Long Lan<sup>1,‡</sup>&nbsp;&nbsp;
    Jing Zhang<sup>6</sup>&nbsp
    Zhiyuan Liu<sup>2</sup> &nbsp;&nbsp;
    Maosong Sun<sup>2</sup>&nbsp;&nbsp;&nbsp;
    </br></br>
  <sup>1</sup> National University of Defense Technology&nbsp;&nbsp;&nbsp;
  <sup>2</sup>Tsinghua University &nbsp;&nbsp;&nbsp;
  <sup>3</sup>Wuhan University&nbsp;&nbsp;</br>
<sup>4</sup>University of Chinese Academic of Sciences&nbsp;&nbsp;
<sup>5</sup>Nankai University&nbsp;&nbsp;
<sup>6</sup>The University of Sydney
  </p>

## Intruduction
-  `RS-4M`: A large-scale remote sensing dataset. This dataset, comprising 4 million optical images, 
is designed to fully leverage the representation learning capabilities of MIM methods in RS applications, distinguished by its diverse scene details.</br>
- `SelectiveMAE`: A novel and efficient MIM method tailored for remote sensing images. This method incorporates a new PSTS module,
which significantly accelerates convergence and enhances representation learning compared to the original MIM approach.


## Todo List
- [x] Initial release of checkpoint of SelectiveMAE. 🚀
- [ ] Codes and configs for downstream tasks of SelectiveMAE, Scene Classification. 🚀
- [ ] Codes and configs for downstream tasks of SelectiveMAE, Object Detection and Semantic Segmentation. 
- [ ] Pretraining codes and configs for SelectiveMAE will be released.
- [ ] RS-4M dataset will be released.



## Updates

- \[2024.06\] - The training logs of the SelectiveMAE have been released.



## Outline

- [RS-4M](#RS-4M)
- [Installation](#gear-installation)
- [Pretraining](#blue_car-Pretraining)
- [Downstream Tasks](#rocket-Results-on-downstream-tasks)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## RS-4M
 `RS-4M` dataset contains about 4 million high-quality remote sensing optical images, which is four times larger than previous representative remote sensing datasets.

### Examples of RS-4M
<img src="./Figures/RS-4M.png" width="700"> 

### Experiments on RS-4M
RS-4M offers a significantly larger and more diverse image set compared to previous datasets. To evaluate its effectiveness, we pre-train a **ViT-Base** model using the vanilla **MAE** method. 
For comparison, we use the [**MillionAID**](https://captain-whu.github.io/DiRS/) dataset, maintaining an equal number of data points during training: 800 epochs for **MillionAID**'s 1 million images and 200 epochs for our **RS-4M** dataset.

|  Dataset   | Pretrained model | Images Number | Epoch | Sence  Classification |    Sence  Classification    |    Object  Detection      |     Object  Detection    | Semantic Segmentation | Semantic Segmentation|
|:----------:|:----------------:|:-------------:|:-----:|:---------------------:|:---------------------------:|:-------------------------:|:-----------------:|:--------:|:------------:|
|            |                  |               |       |          AID          |          RESISC-45          |           DIOR            |      DIOR-R       |  LoveDA  |  SpaceNetv1  |
|            |                  |               |       |  OA (TR=20%/50%)    |       OA (TR=20%/50%)       |           mAP50           |       mAP50       |   mIoU   |      mF1     |
| MillionAID |     [Weights](https://pan.baidu.com/s/1OCl7whWnYoyrAI8zha_Kbg?pwd=0330)      |   1 million   |  800  |      94.92/97.38      |         89.20/93.60         |           71.80           |       62.33       |   51.24  |     79.24    |
|   RS-4M    |     [Weights](https://pan.baidu.com/s/1-6HBRbAyHMUrTSwcSOIhyw?pwd=0330)      |   2 million   |  400  |      96.64/98.10      |         91.80/94.31         |           73.90           |       65.95       |   52.86  |     79.37    |
|   RS-4M    |     [Weights](https://pan.baidu.com/s/1S_oTibDouAi-VrmESn7qPg?pwd=0330)      |   3 million   |  267  |      96.67/98.18      |         92.24/94.41         |           75.40           |       67.07       |   52.39  |     79.37    |
|   RS-4M    |     [Weights](https://pan.baidu.com/s/1zmS24CqFo44Rkkkl2YqeaQ?pwd=0330)      |   4 million   |  200  |      96.10/98.03      |         92.38/94.30         |           74.70           |       66.26       |   52.75  |     79.23    |
|   RS-4M    |     [Weights](https://pan.baidu.com/s/1Qrgtv7Dotfb_QQ2GCk6bog?pwd=0330)      |   4 million   |  800  |      **96.88/98.22**      |         **92.44/94.43**         |           **75.40**           |      **67.35**       |   **52.80**  |    **79.41**    |


## SelectiveMAE

### :gear: Installation

For details related to installation, kindly refer to [INSTALL.md](docs/INSTALL.md).


###  :blue_car:  Pretraining

To learn more usage about the pretraining codes, kindly refer to [PRETRAIN.md](docs/GET_STARTED.md).


### :rocket: Results on downstream tasks

| Model        | Publication |  Backbone  | Sence  Classification | Sence  Classification  |   Object Detection  |      Object Detection        |   Semantic Segmentation   |    Semantic Segmentation         |
|--------------|:-----------:|:----------:|:---------------------:|:-----------------:|:----------:|:----------:|:------------:|:----------:|
|              |             |            |          AID          |     RESISC-45     |    DIOR    |   DIOR-R   |    LoveDA    | SpaceNetv1 |
|              |             |            |    OA (TR=20%/50%)    | OA (TR=20%/50%)   |   mAP50    | mAP50      |     mIoU     |   mF1      |
| SeCo         |   ICCV'21   |  ResNet-50 |      93.47/95.99      |    89.64/92.91    |      -     |      -     |     43.63    |    77.09   |
| GASSL        |   ICCV'21   |  ResNet-50 |      93.55/95.92      |    90.86/93.06    |    67.40   |    65.65   |     48.76    |    78.51   |
| TOV          |  JSTARS'23  |  ResNet-50 |      95.16/97.09      |    90.97/93.79    |    70.16   |    66.33   |     49.70    |      -     |
| CACo         |   CVPR'23   |  ResNet-50 |      90.88/95.05      |    88.28/91.94    |    66.91   |    64.10   |     48.89    |    77.94   |
| SatMAE       |   NIPS'22   |    ViT-L   |      95.02/96.94      |    91.72/94.10    |    70.89   |    65.66   |       -      |    78.07   |
| ScaleMAE     |   ICCV'23   |    ViT-L   |      96.44/97.58      |    92.63/95.04    |    73.81   |    66.47   |       -      |      -     |
| SSL4EO       |   GRSM'23   |    ViT-S   |      91.06/94.74      |    87.60/91.27    |    64.82   |    61.23   |       -      |      -     |
| RingMo       |   TGRS'22   |   Swin-B   |      96.90/98.34      |    94.25/95.67    |    75.90   |      -     |       -      |      -     |
| SatLas       |   ICCV'23   |   Swin-B   |      94.96/97.38      |    92.16/94.70    |    74.10   |    67.59   |       -      |      -     |
| GFM          |   ICCV'23   |   Swin-B   |      95.47/97.09      |    92.73/94.64    |    72.84   |    67.67   |       -      |      -     |
| RVSA         |   TGRS'23   | ViT-B+RVSA |      97.03/98.50      |    93.93/95.69    |    75.80   |    68.06   |     51.95    |      -     |
| [SelectiveMAE](https://pan.baidu.com/s/1Y4WBj35-HAKeZJe125TG8Q?pwd=0330) |      -      |    ViT-B   |      96.78/98.12      |    93.35/94.58    |    75.70   |    67.78   |     53.05    |   **79.50**  |
| [SelectiveMAE ](https://pan.baidu.com/s/1miSlmoeZLjzc_WgXE87Fxg?pwd=0330)|      -      |    ViT-L   |     **97.25/98.48**     |    **94.57/95.77**    |   **77.80**  |    **70.31**   |     **54.31**  |    79.46   |

## License

This work is under the [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0), while some specific operations in this codebase might be with other licenses. Please refer to [LICENSE.md](docs/LICENSE.md) for a more careful check, if you are using our code for commercial matters.

## Acknowledgements
