python class_balanced_linear_prob.py \
  /ssd0/fercus/imagenet \
  --index_name default_lift_supervised_wd_1e_5_pl_thre_25_budget_12 \
  --dataset imagenet127 \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --cos \
  --epochs 300 \
  --warmup-epoch 10 \
  --pretrained checkpoints/supervised/imagenet127_wd_1e_5/11-24-2022-17:36/best_val_model.pth.tar \
  --amp-opt-level O1 \
  --train-type DefaultVal \
  --cls 127 \