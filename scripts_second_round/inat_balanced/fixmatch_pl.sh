python class_balanced_linear_prob.py \
  /ssd0/fercus/inat \
  --index_name default_lift_supervised_wd_1e_5_pl_thre_18_budget_6 \
  --dataset inat \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --cos \
  --epochs 300 \
  --warmup-epoch 10 \
  --pretrained checkpoints/semi_supervised/inat_fixmatch/09-16-2022-11:30/best_val_model.pth.tar \
  --amp-opt-level O1 \
  --train-type DefaultVal \
  --cls 810 \