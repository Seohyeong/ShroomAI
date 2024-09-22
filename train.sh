python ShroomAI/model/train_torch.py \
  --pretrain \
  --finetune \
  --evaluate \
  --pretrain_batch_size 256 \
  --finetune_batch_size 128 \
  --eval_batch_size 512 \
  --pretrain_epoch 30 \
  --finetune_epoch 40