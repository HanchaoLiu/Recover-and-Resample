
# Recovering Complete Actions for Cross-dataset Skeleton Action Recognition 

This is the code for the paper "Recovering Complete Actions for Cross-dataset Skeleton Action Recognition" (NeurIPS 2024).

[**paper**](), [**arxiv**](https://arxiv.org/abs/2410.23641)

[**jittor implementation**]()

<!-- [place overall pipeline figure here.] -->





## Project structure
```
Recover-and-Resample
  |--feeders          
  |--gendata
  |--main
  |--main_others
  |--scripts
  |--scripts_others
  |--nets
  |--configs                   
  |--config.py
```

## Data preparation 

We provide scripts to process data for our 18-class cross-dataset setting using [NTU-60](https://rose1.ntu.edu.sg/dataset/actionRecognition/), [ETRI](https://nanum.etri.re.kr/share/judekim/HMI?lang=En_us), and [PKU-MMD-v1](https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html), and 51-class cross-dataset setting using NTU-60 and PKU-MMD-v1. Please refer to 

```
gendata/readme_gendata.md
```

Move processed data to ./data, so that the directories will be like
```
data
  |--common18
     |--xsub64
        |--etri       
        |--etriA       
        |--etriE       
        |--ntu       
        |--pku  
  |--ntupku51
     |--xsub64
        |--ntu
        |--pku   
```

Create directories to save training results.
```
mkdir -p workdir/ws
mkdir -p workdir/result
```


## Boundary pose and linear transform clustering
Perform boundary pose clustering and linear transform clustering. 
```
sh scripts/do_clustering_T.sh
```

The clustering results will be saved to `./data/cluster_result` and will be used by our recovering stage. We also provide the necessary cluster result files under `./data/cluster_result`.

## Training and testing

#### For the 18-class cross-dataset setting (N->E, N->P, E_A->N, E_A->P) with AGCN
Ours 


```
sh scripts/train_3domain_ours.sh 0
```

ERM
```
sh scripts/train_3domain_erm.sh 0
```

#### For the 51-class cross-dataset setting (N51->P51, P51->N51) with HCN
Ours 


```
sh scripts/train_ntupku51_ours_N51.sh 0
```



```
sh scripts/train_ntupku51_ours_P51.sh 0
```



<!-- [checkpoint or training log.] -->


#### Implementation for some other baseline methods

ST-Cubism

```
sh scripts_others/train_tjigsaw.sh 0
sh scripts_others/train_sjigsaw.sh 0
```

```
sh scripts_others/train_tjigsaw_ntupku51_N51.sh 0
sh scripts_others/train_sjigsaw_ntupku51_N51.sh 0
```

TODO: clean up code for some other baseline methods.


## Acknowledgements

Our code is mainly built on [2s-AGCN](https://github.com/lshiwjx/2s-AGCN). 

Code for baseline methods partially comes from 
[ST-Cubism](https://github.com/shanice-l/st-cubism), 
[skeleton adversarial attack](https://github.com/realcrane/Understanding-the-Robustness-of-Skeleton-based-Action-Recognition-under-Adversarial-Attack), 
[CoDT](https://github.com/Qinying-Liu/CoDT), 
[HICLR](https://github.com/JHang2020/HiCLR), 
[Uniform sampling](https://github.com/kennymckormick/pyskl), 
[CropPad](https://github.com/LinguoLi/CrosSCLR), 
[CropResize](https://github.com/fmthoker/skeleton-contrast), 
[OTAM](https://github.com/wangzehui20/OTAM-Video-via-Temporal-Alignment), 
[HCN](https://github.com/huguyuehuhu/HCN-pytorch), 
[ST-GCN](https://github.com/yysijie/st-gcn), 
[CTR-GCN](https://github.com/Uason-Chen/CTR-GCN).

We thank them for kindly releasing their code.

#### Bibtex
If you find this code useful in your research, please consider citing:

```
@inproceedings{liu2024recovering,
title={Recovering Complete Actions for Cross-dataset Skeleton Action Recognition},
author={Liu, Hanchao and Li, Yujiang and Mu, Tai-Jiang and Hu, Shi-Min},
booktitle={NeurIPS},
year={2024}
}
```


## License
This code is distributed under an [MIT LICENSE](LICENSE).



