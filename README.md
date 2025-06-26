
<p align="center">

  <h2 align="center"><strong>	Harnessing Massive Satellite Imagery with Efficient Masked Image Modeling</strong></h2>

  <p align="center">
     Fengxiang Wang<sup>1</sup>&nbsp;&nbsp;&nbsp;
    Hongzhen Wang<sup>2,‡</sup>&nbsp;&nbsp;&nbsp;
    Di Wang<sup>3</sup>&nbsp;&nbsp;&nbsp;
    Zonghao Guo<sup>2</sup></br>
    Zhenyu Zhong<sup>4</sup>&nbsp;&nbsp;&nbsp;
    Long Lan<sup>1,‡</sup>&nbsp;&nbsp;
    Wenjing Yang<sup>1,‡</sup>&nbsp;&nbsp;&nbsp;
    Jing Zhang<sup>3,‡</sup>&nbsp
    </br></br>
  <sup>1</sup> National University of Defense Technology&nbsp;&nbsp;&nbsp;
  <sup>2</sup>Tsinghua University &nbsp;&nbsp;&nbsp;
  <sup>3</sup>Wuhan University&nbsp;&nbsp;
<sup>4</sup>Nankai University&nbsp;&nbsp;
<div align='center' style="font-size: larger; "><strong>ICCV 2025</strong></div>

  <p align="center">
    📃 <a href="https://arxiv.org/abs/2406.11933" target="_blank">Paper</a> |
    🤗 <a href="https://huggingface.co/datasets/initiacms/OpticalRS-4M" target="_blank"> OpticalRS-4M</a> |
    🤗 <a href="https://huggingface.co/datasets/initiacms/OpticalRS-13M" target="_blank"> OpticalRS-13M</a> |
    🤗 <a href="https://huggingface.co/initiacms/SelectiveMAE" target="_blank">Models</a>
  </p>
  </p>

## 🎯Intruduction
-  `Dataset`: `OpticalRS-13M` is a large-scale remote sensing dataset. This dataset, comprising 13 million optical images, 
is designed to fully leverage the representation learning capabilities of MIM methods in RS applications, distinguished by its diverse scene details. We also offer a light version, named `OpticalRS-4M`.</br>
- `SelectiveMAE`: A novel and efficient MIM method tailored for remote sensing images. This method incorporates a new PSTS module,
which significantly accelerates convergence and enhances representation learning compared to the original MIM approach.


## ✅ Todo List
- [x] Initial release of checkpoint of SelectiveMAE. 
- [x] Pretraining codes and configs for SelectiveMAE have be released. 
- [x] OpticalRS-4M dataset has be released. 
- [x] OpticalRS-13M dataset will be released. 
- [ ] Codes and configs for downstream tasks of Scene Classification. 
- [ ] Codes and configs for downstream tasks of Object Detection and Semantic Segmentation.



## 🔥 News
- \[2025.06\] - **SelectiveMAE has been accepted by ICCV2025.**
- \[2025.06\] - OpticalRS-13M has been released on 🤗[HuggingFace](https://huggingface.co/datasets/initiacms/OpticalRS-13M).
- \[2025.06\] - Models have been released on 🤗[HuggingFace](https://huggingface.co/initiacms/SelectiveMAE).
- \[2025.06\] - OpticalRS-4M has been released on 🤗[HuggingFace](https://huggingface.co/datasets/initiacms/OpticalRS-4M).
- \[2025.06\] - The pretraining codes of the SelectiveMAE have been released.
- \[2024.06\] - Paper has been released on [arxiv](https://arxiv.org/abs/2406.11933).
- \[2024.06\] - The training logs and checkpoints of the SelectiveMAE have been released.



## 📚 Contents

- [OpticalRS-4M](#opticalrs-4m)
- [OpticalRS-13M](#opticalrs-13m)
- [SelectiveMAE](#selectivemae)
- [Citation](#citation)
- [License](#license)


## 🚀OpticalRS-4M
### Usage
`OpticalRS-4M` available on 🤗HuggingFace via [OpticalRS-4M](https://huggingface.co/datasets/initiacms/OpticalRS-4M).

Use the following command to unzip:
```bash
# if 7z is available
7z x OpticalRS-4M.zip
# if zip and unzip is available
zip -s 0 OpticalRS-4M.zip --out whole.zip
unzip whole.zip
```

### Experiments on OpticalRS-4M
OpticalRS-4M offers a significantly larger and more diverse image set compared to previous datasets. To evaluate its effectiveness, we pre-train a **ViT-Base** model using the vanilla **MAE** method. 
For comparison, we use the [**MillionAID**](https://captain-whu.github.io/DiRS/) dataset, maintaining an equal number of data points during training: 800 epochs for **MillionAID**'s 1 million images and 200 epochs for our **OpticalRS-4M** dataset.

|  Dataset   | Pretrained model | Images Number | Epoch | Sence  Classification |    Sence  Classification    |    Object  Detection      |     Object  Detection    | Semantic Segmentation | Semantic Segmentation|
|:----------:|:----------------:|:-------------:|:-----:|:---------------------:|:---------------------------:|:-------------------------:|:-----------------:|:--------:|:------------:|
|            |                  |               |       |          AID          |          RESISC-45          |           DIOR            |      DIOR-R       |  LoveDA  |  SpaceNetv1  |
|            |                  |               |       |  OA (TR=20%/50%)    |       OA (TR=20%/50%)       |           mAP50           |       mAP50       |   mIoU   |      mF1     |
| MillionAID |     [Weights](https://pan.baidu.com/s/1OCl7whWnYoyrAI8zha_Kbg?pwd=0330)      |   1 million   |  800  |      94.92/97.38      |         89.20/93.60         |           71.80           |       62.33       |   51.24  |     79.24    |
|   OpticalRS-4M    |     [Weights](https://pan.baidu.com/s/1-6HBRbAyHMUrTSwcSOIhyw?pwd=0330)      |   2 million   |  400  |      96.64/98.10      |         91.80/94.31         |           73.90           |       65.95       |   52.86  |     79.37    |
|   OpticalRS-4M    |     [Weights](https://pan.baidu.com/s/1S_oTibDouAi-VrmESn7qPg?pwd=0330)      |   3 million   |  267  |      96.67/98.18      |         92.24/94.41         |           75.40           |       67.07       |   52.39  |     79.37    |
|   OpticalRS-4M    |     [Weights](https://pan.baidu.com/s/1zmS24CqFo44Rkkkl2YqeaQ?pwd=0330)      |   4 million   |  200  |      96.10/98.03      |         92.38/94.30         |           74.70           |       66.26       |   52.75  |     79.23    |
|   OpticalRS-4M    |     [Weights](https://pan.baidu.com/s/1Qrgtv7Dotfb_QQ2GCk6bog?pwd=0330)      |   4 million   |  800  |      **96.88/98.22**      |         **92.44/94.43**         |           **75.40**           |      **67.35**       |   **52.80**  |    **79.41**    |


## 🚀OpticalRS-13M
`OpticalRS-13M` available on 🤗HuggingFace via [OpticalRS-13M](https://huggingface.co/datasets/initiacms/OpticalRS-13M). Follow OpticalRS-4M to unzip.

## 🚀SelectiveMAE

### :gear: Installation for Pretraining
Please install the pretraining dependencies in `SelectiveMAE/requirements.txt`:
```sh
# Optionally create a conda environment
conda create -n selectivemae python=3.10 -y
conda activate selectivemae
# Install dependencies
pip install -r requirements.txt
```

###  :blue_car:  Pretraining for SelectiveMAE

To pre-train ViT-Base, run the following on 8 GPUs:
```sh
torchrun --nproc_per_node=8 --nnodes 1 --master_port 16666 main_pretrain.py --batch_size 256 --selectivemae --dataset opticalrs-4m --dataset_path 'your_dataset_path' --model mae_vit_base_patch16 --output_dir output --norm_pix_loss --blr 1.5e-4 --weight_decay 0.05  --num_workers 12  --decoder_depth 12 --mask_ratio 0.85 --kept_mask_ratio 0.25 --epochs 800 --warmup_epochs 30
```
First, download the corresponding dataset, then set `opticalrs-4m` or `opticalrs-13m`, and update the dataset path accordingly. To train ViT-Small or ViT-Large, set `--model mae_vit_small_patch16` or `--model mae_vit_large_patch16`. You can use `--accum_iter` to perform gradient accumulation if your hardware could not fit the batch size. [FlashAttention 2](https://github.com/Dao-AILab/flash-attention) should be installed with `pip install flash-attn --no-build-isolation`.


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
| SelectiveMAE(OpticalRS-4M) |      [Baidu](https://pan.baidu.com/s/1Y4WBj35-HAKeZJe125TG8Q?pwd=0330) & [HuggingFace](https://huggingface.co/initiacms/SelectiveMAE)     |    ViT-B   |      96.90/98.12      |    93.35/94.58    |    75.70   |    67.78   |     53.05    |   **79.50**  |
| SelectiveMAE(OpticalRS-4M)|     [Baidu](https://pan.baidu.com/s/1miSlmoeZLjzc_WgXE87Fxg?pwd=0330) & [HuggingFace](https://huggingface.co/initiacms/SelectiveMAE)      |    ViT-L   |     97.25/98.48    |    94.57/95.77    |   77.80  |    70.31  |     **54.31**  |    79.46   |
| SelectiveMAE(OpticalRS-13M) |      [Baidu](https://pan.baidu.com/s/1_QNLBGhrViapquDcVZHnkw?pwd=bmzj) & [HuggingFace](https://huggingface.co/initiacms/SelectiveMAE)     |    ViT-B   |     97.10/98.28     |    93.70/95.48   |   75.80   |    67.69  |     52.68    |   79.44  |
| SelectiveMAE(OpticalRS-13M)|     [Baidu](https://pan.baidu.com/s/10HJ_kZwW2nxNqDNjJRb6SQ?pwd=eyjn) & [HuggingFace](https://huggingface.co/initiacms/SelectiveMAE)     |    ViT-L   |    **97.49/98.52**   |   **94.73/96.36**   |   **78.70**  |   **71.75**  |    53.92  |    79.48  |

# 🔗Citation
If you find SelectiveMAE helpful, please consider citing:

```latex
@article{wang2024scaling,
  title={Scaling efficient masked autoencoder learning on large remote sensing dataset},
  author={Wang, Fengxiang and Wang, Hongzhen and Wang, Di and Guo, Zonghao and Zhong, Zhenyu and Lan, Long and Zhang, Jing and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv e-prints},
  pages={arXiv--2406},
  year={2024}
}
```

## 🤝License

This work is under the [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0), while some specific operations in this codebase might be with other licenses. Please refer to [LICENSE.md](docs/LICENSE.md) for a more careful check, if you are using our code for commercial matters.
