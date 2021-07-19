# The baselines

This directory, ```/baselines```, is the baseline methods compared with in our paper titled "AutoDebias: Learning to Debias for Recommendation". 

## The steps to run the baselines methods:

To run the baselines, two steps are required: 

1. Move the baseline files to their parent directory. 
2. Excute the command to run the baselines. 

**An example to run the baseline:**  

To run model MF(biased) on dataset Yahoo!R3 (supposing that the current directory is AutoDebias/), two steps are as follows: 

```
mv ./baselines/MF_bias.py . 
```
```
python MF_bias.py --dataset yahooR3
```

## Correspondence between baseline methods and files 

In our paper, there are three main tables in regard to explicit feedback (Table 3), implicit feedback (Table 7), and list feedback (Table 8). Here the correspondence between baseline methods and files in the three tables are given: 

**Explicit feedback (Table 3):**

|   Method    |     File      |
| :---------: | :-----------: |
| MF(biased)  |  MF_bias.py   |
| MF(uniform) | MF_uniform.py |
| MF(combine) | MF_combine.py |
|     IPS     |    IPS.py     |
|     DR      |     DR.py     |
|    CausE    |   CausE.py    |
|  KD-Label   |  KD_Label.py  |

**Implicit feedback (Table 7):**

| Method |  File  |
| :----: | :----: |
|  WMF   | WMF.py |
| Rl-MF  |  ---   |
|  AWMF  |  ---   |

For the RI-MF and AWMF methods, we use the codes released by the authors of the two methods. 

**List feedback (Table 8):**

|   Method   |    File    |
| :--------: | :--------: |
| MF(biased) | MF_bias.py |
|    DLA     |   DLA.py   |
|   HeckE    |  HeckE.py  |

