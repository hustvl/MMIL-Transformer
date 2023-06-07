# Multi-level Multiple Instance Learning with Transformer for Whole Slide Image Classification

## Project Info
This project is the official implementation of MMIL-Transformer proposed in paper [Multi-level Multiple Instance Learning with Transformer for Whole Slide Image Classification]

## Prerequisites

* Python 3.8.10
* Pytorch 1.12.1
* torchmetrics 0.4.1
* CUDA 11.6
* numpy 1.24.2
* einops 0.6.0
* sklearn 1.2.2
* h5py 3.8.0
* pandas 2.0.0
* nystrom_attention
* argparse

## Pretrained Weight
All test experiments were conducted 10 times to calculate the average ACC and AUC.
<div align="center">

| model name | grouping method | weight | ACC | AUC |
|------------|-----|:------:|----|----|
| `TCGA_embed`|Embedding grouping|[HF link](https://huggingface.co/RJKiseki/MMIL-Transformrt/blob/main/TCGA_embed.pt) | 93.15% | 98.97% |
| `TCGA_random`|Random grouping|[HF link](https://huggingface.co/RJKiseki/MMIL-Transformrt/blob/main/TCGA_random.pt) | 94.37%| 99.04% |
| `TCGA_random_with_subbags_0.75masked`|Random grouping + mask|[HF link](https://huggingface.co/RJKiseki/MMIL-Transformrt/blob/main/TCGA_random_mask_0.75.pt) | 93.95%| 99.02% |
| `camelyon16_random`|Random grouping|[HF link](https://huggingface.co/RJKiseki/MMIL-Transformrt/blob/main/camelyon16_random.pt) | 91.78% | 94.07% |
| `camelyon16_random_with_subbags_0.6masked`| Random grouping + mask|[HF link](https://huggingface.co/RJKiseki/MMIL-Transformrt/blob/main/camelyon16_mask_0.6.pt) | 93.41% | 94.74% |
</div>


## Usage
  ### Dataset

    #### TCGA Dataset
We use the same configuration of data preprocessing as [DSMIL](https://github.com/binli123/dsmil-wsi). Or you can directly download the feature vector they provided for TCGA.

#### CAMELYON16 Dataset
We use [CLAM](https://github.com/mahmoodlab/CLAM/tree/master) to preprocess CAMELYON16 at 20x.

#### Preprocessed feature vector
Preprocess WSI is time consuming and difficult. We also provide processed feature vector for two datasets. Aforementioned works [DSMIL](https://github.com/binli123/dsmil-wsi) and [CLAM](https://github.com/mahmoodlab/CLAM/tree/master) 
greatly simplified the preprocessing. Thanks again to their wonderful works!
<div align="center">
  
| Dataset | Link | Disk usage |
|------------|:-----:|----|
| `TCGA`|[HF link](https://huggingface.co/datasets/RJKiseki/TCGA)| 16GB |
| `CAMELYON16`|[HF link](https://huggingface.co/datasets/RJKiseki/CAMELYON16)|20GB|
</div>


### Test the model

For TCGA testing:
```
python main.py \
--test {Your_Path_to_Pretrain} \
--num_test 10 \
--type TCGA \
--num_subbags 4 \
--mode {embed or random} \
--num_msg 1 \
--num_layers 2 \
--csv {Your_Path_to_TCGA_csv} \
--h5 {Your_Path_to_h5_file}
```


For CAMELYON16 testing:
```
python main.py \
--test {Your_Path_to_Pretrain} \
--num_test 10 \
--type camelyon16 \
--num_subbags 10 \
--mode random \
--num_msg 1 \
--num_layers 2 \
--csv {Your_Path_to_CAMELYON16_csv}\
--h5 {Your_Path_to_h5_file}
```
