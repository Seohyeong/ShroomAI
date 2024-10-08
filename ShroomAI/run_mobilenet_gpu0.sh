# no multitask learning, mobilenet
CUDA_VISIBLE_DEVICES=0, python ShroomAI/run.py \
  --dataset_dir_path '/home/user/seohyeong/ShroomAI/ShroomAI/dataset/inat_300' \
  --pretrain \
  --finetune \
  --model_name 'mobilenet_v2' \
  --img_size 224 \
  --pt_bs 1024 \
  --ft_bs 256 \
  --eval_bs 1024 \
  --pt_epoch 20 \
  --ft_epoch 30 \
  --pt_lr 0.0005 \
  --ft_lr 0.00001

# multitask learning, mobilenet
CUDA_VISIBLE_DEVICES=0, python ShroomAI/run.py \
  --dataset_dir_path '/home/user/seohyeong/ShroomAI/ShroomAI/dataset/inat_300' \
  --pretrain \
  --finetune \
  --model_name 'mobilenet_v2' \
  --img_size 224 \
  --pt_bs 1024 \
  --ft_bs 256 \
  --eval_bs 1024 \
  --pt_epoch 20 \
  --ft_epoch 30 \
  --pt_lr 0.0005 \
  --ft_lr 0.00001 \
  --meta_info_path '/home/user/seohyeong/ShroomAI/ShroomAI/dataset/meta.json'