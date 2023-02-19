python class_balanced_linear_prob.py \
  /ssd0/fercus/inat \
  --index_name default \
  --dataset inat \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --cos \
  --epochs 300 \
  --warmup-epoch 10 \
  --pretrained checkpoints/semi_supervised/inat_debiased_tau_4/10-09-2022-13:24/best_val_model.pth.tar \
  --amp-opt-level O1 \
  --train-type DefaultVal \
  --cls 810 \