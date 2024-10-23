
### Prepare raw data

Ask for permission and download datasets [NTU-RGBD-60](), [ETRI-Activity3D](), and [PKU-MMD-v1]().
Copy raw data to Recover-and-Resample/data/raw_data

For NTU, (decompressed from nturgb+d_skeletons.zip)

```
raw_data/nturgbd_raw/nturgb+d_skeletons
raw_data/nturgbd_raw/samples_with_missing_skeletons.txt
```

For ETRI, (decompressed from Skeleton(P001-P050).zip and Skeleton(P051-P100).zip)

```
raw_data/etri_data/P001-P050
raw_data/etri_data/P051-P100
```

For pku mmd v1

```
raw_data/pkummd_v1
```

which has the structure like 
pkummd_v1/Data/PKU_Skeleton_Renew/0002-L.txt, pkummd_v1/Label/Train_Label_PKU_final/0002-L.txt and have folders
(Data, Label, Split)


### gendata_3domain (18-class)

1. generate full 300 pre-normed train/test data and label for each domain. using `gendata_{domain}.py`

```
python gendata_etri.py 
python gendata_pku.py 
python gendata_ntu.py
```

so that we can have the following data in ../data/processed_data
```
domain
  |- train_data.npy
  |- train_label.pkl
  |- test_data.npy 
  |- test_label.pkl
```

2. select common18 data
```
python gen_common18.py --domain pku
python gen_common18.py --domain ntu 
python gen_common18.py --domain etriA
python gen_common18.py --domain etriE
```

This will save data to ../data/processed_data/common18/xsub300


3. downsample sequence length from 300 to 64 

```
python downsample_motion.py --input_dir ../data/processed_data/common18/xsub300/pku --output_dir ../data/processed_data/common18/xsub64/pku

python downsample_motion.py --input_dir ../data/processed_data/common18/xsub300/ntu --output_dir ../data/processed_data/common18/xsub64/ntu

python downsample_motion.py --input_dir ../data/processed_data/common18/xsub300/etriA --output_dir ../data/processed_data/common18/xsub64/etriA

python downsample_motion.py --input_dir ../data/processed_data/common18/xsub300/etriE --output_dir ../data/processed_data/common18/xsub64/etriE
```

This will save data to ../data/processed_data/common18/xsub64


4. merge etriA and etriE for test set.
```
python merge_adult_elder.py
```


Move ../data/processed_data/common18/xsub64 to ../data so that we have 
Recover-and-Resample/data/common18/xsub64
```
mv ../data/processed_data/common18 ../data/
```



### gendata ntu pku 51 (51-class)

Run
```
python gendata_pku.py 
python gendata_ntu.py
```

so that we can have the following data in ../data/processed_data
```
domain
  |- train_data.npy
  |- train_label.pkl
  |- test_data.npy 
  |- test_label.pkl
```

Then, run
```
python gen_ntu_pku_51.py 
```

This will save data to ../data/processed_data/ntupku51/xsub64

Move ../data/processed_data/ntupku51/xsub64 to ../data so that we have 
Recover-and-Resample/data/ntupku51/xsub64
```
mv ../data/processed_data/ntupku51 ../data/
```


After the above steps, the generated data will be like

```
Recover-and-Resample
|
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