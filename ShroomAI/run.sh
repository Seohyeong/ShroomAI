CUDA_VISIBLE_DEVICES=0, python ShroomAI/run.py \
  --pretrain \
  --finetune \
  --pt_bs 1024 \
  --ft_bs 128 \
  --eval_bs 1024 \
  --pt_epoch 30 \
  --ft_epoch 40