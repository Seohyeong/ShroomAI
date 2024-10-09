# ShroomAI
Mushroom classification using a dataset scraped from [GBIF (Global Biodiversity Information Facility)](https://www.gbif.org/). 

🍄 Check out the [app version](https://github.com/Seohyeong/ShroomScanner/tree/main)!

## About the Dataset
This dataset consists of mushroom images (iNaturalist Research-grade Observations, between start of 2000 and end of 2024) collected from [GBIF (Global Biodiversity Information Facility)](https://www.gbif.org/). 
Each image is associated with a specific species, selected from a hierarchical taxonomy that includes multiple layers such as family, phylum, class, order, and species. `Species` is used as labels for the images in the dataset.

For further details, refer to `preprocess/preprocess.ipynb` and `preprocess/get_images.py`.

**Dataset Details**  
Total Species: 1,000   
Images per Species (train): ~300   
Images per Species (val): ~30 


### Citation
```
GBIF.org (02 October 2024) GBIF Occurrence Download  https://doi.org/10.15468/dl.mu3ech
```

### Examples
These are 16 image/label pairs from the collected dataset.

<img src="readme_docs/examples.png" width="700" >


## Results
| Backbone         | # params   | acc           |  ckpt |
|------------------|------------|---------------|-------|
| mobilenet_v2     | 3,504,872  |      |  |
| efficientnet_b0  | 5,288,548  |      |  |


## How to Train
```bash
# example
CUDA_VISIBLE_DEVICES=0, python ShroomAI/run.py \
  --dataset_dir_path {path_to_dataset} \
  --pretrain \
  --finetune \
  --model_name {'mobilenet_v2' or 'efficientnet_b0'} \
  --img_size 224 \
  --pt_bs 1024 \
  --ft_bs 256 \
  --eval_bs 1024 \
  --pt_epoch 20 \
  --ft_epoch 30 \
  --pt_lr 0.0005 \
  --ft_lr 0.00001
```


