# no multitask learning, mobilenet
CUDA_VISIBLE_DEVICES=0, python ShroomAI/run.py \
  --pretrain \
  --finetune \
  --model_name 'mobilenet_v2' \
  --pt_bs 1024 \
  --ft_bs 256 \
  --eval_bs 1024 \
  --pt_epoch 20 \
  --ft_epoch 30 \
  --pt_lr 0.0005 \
  --ft_lr 0.00001

# multitask learning, mobilenet
CUDA_VISIBLE_DEVICES=1, python ShroomAI/run.py \
  --pretrain \
  --finetune \
  --model_name 'mobilenet_v2' \
  --pt_bs 1024 \
  --ft_bs 256 \
  --eval_bs 1024 \
  --pt_epoch 20 \
  --ft_epoch 30 \
  --pt_lr 0.0005 \
  --ft_lr 0.00001 \
  --meta_info_path '/home/user/seohyeong/ShroomAI/ShroomAI/dataset/images_100_3314_meta.json'

# multitask learning, efficientnet
CUDA_VISIBLE_DEVICES=0, python ShroomAI/run.py \
  --pretrain \
  --finetune \
  --model_name 'efficientnet_b0' \
  --pt_bs 1024 \
  --ft_bs 256 \
  --eval_bs 1024 \
  --pt_epoch 20 \
  --ft_epoch 30 \
  --pt_lr 0.01 \
  --ft_lr 0.00005 \
  --meta_info_path '/home/user/seohyeong/ShroomAI/ShroomAI/dataset/images_100_3314_meta.json'