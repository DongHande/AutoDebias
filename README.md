# AutoDebias

This is the official pytorch implementation of AutoDebias, a debiasing method for recommendation system. AutoDebias is proposed in the paper:

[AutoDebias: Learning to Debias for Recommendation](https://arxiv.org/abs/2105.04170)

by  Jiawei Chen, Hande Dong, Yang Qiu, Xiangnan He, Xin Xin, Liang Chen, Guli Lin and Keping Yang

Published at SIGIR 2021. 

## Introduction

AutoDebias is an automatic debiasing method for recommendation system based on meta learning, exploiting a small amout of uniform data to learn de-biasing parameters and using these parameters to guide the learning of the recommendation model. 

## Environment Requirement

The code runs well under python 3.8.5. The required packages are as follows:

- pytorch == 1.4.0
- numpy == 1.19.1
- scipy == 1.5.2
- pandas == 1.1.3
- cppimport == 20.8.4.2

## Datasets

We use two public datasets (Yahoo!R3 and Coat) and a synthetic dataset (Simulation). 

- user.txt: biased data collected by normal policy of recommendation platform. For Yahoo!R3 and Coat, each line is user ID, item ID, rating of the user to the item. For Simulation, each line is user ID, item ID, position of the item, binary rating of the user to the item. 
- random.txt: unbiased data collected by stochastic policy where items are assigned to users randomly. Each line in the file is user ID, item ID, rating of the user to the item. 

## Run the Code

**Explicit feedback**

- For dataset Yahoo!R3:

```shell
python train_explicit.py --dataset yahooR3
```

- For dataset Coat:

```shell
python train_explicit.py --dataset coat
```

**Implicit feedback**

- For dataset Yahoo!R3:

```shell
python train_implicit.py --dataset yahooR3
```

- For dataset Coat:

```shell
python train_implicit.py --dataset coat
```

**Feedback on list recommendation**

- For dataset Simulation:

```shell
python train_list.py --dataset simulation
```

## Contact

Please contact cjwustc@ustc.edu.cn or [donghd@mail.ustc.edu.cn](mailto:donghd@mail.ustc.edu.cn) if you have any questions about the code and paper.

