
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
- [x] Codes and configs for downstream tasks of SelectiveMAE, Scene Classification. 🚀
- [ ] Codes and configs for downstream tasks of SelectiveMAE, Object Detection and Semantic Segmentation. 
- [ ] Pretraining codes and configs for SelectiveMAE will be released.
- [ ] RS-4M dataset will be released.



## Updates

- \[2024.06\] - The training logs of the SelectiveMAE and the codes of classification in fintuning have been released.



## Outline

- [RS-4M](#bar_chart-benchmark-definition)
- [Installation](#gear-installation)
- [Data Preparation](#hotsprings-data-preparation)
- [Getting Started](#rocket-getting-started)
- [Benchmark](#golf-benchmark)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## RS-4M
 `RS-4M` dataset containing about 4 million high-quality remote sensing optical images, which is four times larger than previous representative remote sensing datasets.

### Examples of RS-4M
<img src="./Figures/RS-4M.png" width="700"> 
