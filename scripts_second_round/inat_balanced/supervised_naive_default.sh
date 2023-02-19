python linear_prob.py \
  /ssd0/fercus/inat \
  --index_name default \
  --dataset inat \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --cos \
  --epochs 300 \
  --warmup-epoch 10 \
  --pretrained checkpoints/supervised/inat_wd_1e_5/11-23-2022-17:26/best_val_model.pth.tar \
  --amp-opt-level O1 \
  --train-type DefaultVal \
  --cls 810 \