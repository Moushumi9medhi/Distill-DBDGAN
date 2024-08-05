# [ACM TOMM (2022)] Distill-DBDGAN: Knowledge Distillation and Adversarial Learning Framework for Defocus Blur Detection

<p align="center">
<img src="assets/pipeline.png" width=100% height=100% 
class="center">
</p>

This repository represents the official implementation of the paper titled "Distill-DBDGAN: Knowledge Distillation and Adversarial Learning Framework for Defocus Blur Detection".

[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b)](https://dl.acm.org/doi/pdf/10.1145/3557897)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Sankaraganesh Jonna](https://www.linkedin.com/in/ganeshjonna/),
Moushumi Medhi,
[Rajiv Ranjan Sahay](https://www.iitkgp.ac.in/department/EE/faculty/ee-rajiv)

We present Distill-DBDGAN, a defocus blur detection model, that segment the blurred regions from a given image affected by defocus
blur in resource-constraint devices. Its core principle is to leverage knowledge distillation by transferring information from the larger teacher network to a compact student network for mobile applicability. All the networks are adversarially trained in an end-to-end manner.

## Datasets
We have included the publicly available datasets [```CUHK```](http://shijianping.me/jnb/index.html), [```DUT```](http://ice.dlut.edu.cn/ZhaoWenda/BTBCRLNet.html), and [```SZU-Blur Detection```](https://github.com/mathxlsun/BD_EFPN/tree/master/SZU-BD) for evaluation. You can find them in the folders [```CUHK_test```], [```DUT_test```], and [```SZU-BD_test```], respectively. If you want to do the inference on your own dataset, you can change the format of your dataset according to the provided dataset. The defocus blur detection masks output by our model are labeled with non-zero values for blur. However, some datasets provide ground truth blur masks where blur is marked as zero. To ensure consistency with our model's output, we need to convert the ground truth masks from these datasets to a uniform format that aligns with our model's format.

## Testing

 To test our model, run the testing script: : ```DISTILL_DBD.sh```

 ```
bash DISTILL_DBD.sh
 ```

## Results
We provide results on all the three datasets: ```CUHK```, ```DUT```, and ```SZU-Blur Detection```. The results obtained from our trained models can be directly accessed from ```Results/dataset_name/```;
<p align="center">
  <img src="assets/CUHK_fm_curves.png" width="45%" style="display: inline-block;">
  <img src="assets/CUHK_pr_curves.png" width="45%" style="display: inline-block;">
  <br>
</p>
<p align="center">
  <img src="assets/DUT_fm_curves.png" width="45%" style="display: inline-block;">
  <img src="assets/DUT_pr_curves.png" width="45%" style="display: inline-block;">
  <br>
</p>
<p align="center">
  <img src="assets/SZU-BD_fm_curves.png" width="45%" style="display: inline-block;">
  <img src="assets/SZU-BD_pr_curves.png" width="45%" style="display: inline-block;">
  <br>
</p>

## Contact

- Moushumi Medhi: medhi.moushumi@iitkgp.ac.in

## Dedication
This paper is for you, [Sankar](https://www.linkedin.com/in/ganeshjonna/) for being an enthusiastic contributor, a dedicated researcher, a genuine friend, and most importantly, an amazing human being. Your expertise was essential in developing the code. May you continue to shine wherever you are. Your influence will always be remembered and cherished.

## Citation
If you find this repo useful, please consider citing:
```
@article{10.1145/3557897,
author = {Jonna, Sankaraganesh and Medhi, Moushumi and Sahay, Rajiv Ranjan},
title = {Distill-DBDGAN: Knowledge Distillation and Adversarial Learning Framework for Defocus Blur Detection},
year = {2023},
issue_date = {April 2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {19},
number = {2s},
issn = {1551-6857},
url = {https://doi.org/10.1145/3557897},
doi = {10.1145/3557897},
journal = {ACM Trans. Multimedia Comput. Commun. Appl.},
month = {feb},
articleno = {87},
numpages = {26}
}
```
## Acknowledgement

We sincerely thank the  owners of the following source code repos: [CUHK](http://shijianping.me/jnb/index.html), [DUT](http://ice.dlut.edu.cn/ZhaoWenda/BTBCRLNet.html), and [SZU-Blur Detection](https://github.com/mathxlsun/BD_EFPN/tree/master/SZU-BD) for sharing the datasets, which are used by our released code.

For the encoder structure of the student network, We utilize the pytorch code: [EfficientNetB3](https://github.com/zhoudaxia233/EfficientUnet-PyTorch/tree/master)

We also thank the anonymous reviewers of ACM TOMM for their helpful feedbacks.
