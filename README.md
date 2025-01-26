# COMP9491-project

This repository contains the code base for my project in COMP9491 Applied AI.

We experimented the following models on the prediction of pedestrians and vehicles on a singnalized intersection:

- [KI-GAN: Knowledge-Informed Generative Adversarial Networks for Enhanced Multi-Vehicle Trajectory Forecasting at Signalized Intersections](https://github.com/ChuhengWei/KI_GAN)

- [Dynamic Neural Relational Inference](https://github.com/cgraber/cvpr_dNRI)

We used the [SinD](https://github.com/SOTIF-AVLab/SinD) dataset in our experiments. The dataset contains positions of traffic participants (vehicle/pedestrians) and traffic light states at different timestamps.

## Installation

Install PyTorch with cuda, and install dependencies listed in `requirements.txt`. See [SinD](SinD) for instruction on downloading the dataset.

Run the below code.

```
pip install -e SinD
pip install -e KI_GAN
pip install -e NRI
```
