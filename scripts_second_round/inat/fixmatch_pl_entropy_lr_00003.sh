# Best Val Acc 0.6550604701042175 @ epoch 257
# Best Test Acc 0.7082942128181458 @ Best val epoch 257
# Best Test Acc 0.7086492776870728 @ epoch 119
# checkpoint saved in:  checkpoints/supervised/inat_default_lift_fixmatch_wd_1e_4_pl_entropy_thre_18_budget_6_lr_00003_wd_1e_5/12-10-2022-23:17
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_supervised.py \
  /scratch/fercus/inat \
  --index_name default_lift_fixmatch_wd_1e_4_pl_entropy_thre_18_budget_6 \
  --dataset inat \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --lr 0.0003 \
  --weight-decay 1e-5 \
  --cos \
  --epochs 300 \
  --warmup-epoch 10 \
  --dist-url 'tcp://localhost:11021' --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained checkpoints/semi_supervised/inat_fixmatch/09-16-2022-11:30/best_val_model.pth.tar \
  --amp-opt-level O1 \
  --train-type RandAugment \
  --output checkpoints/supervised/inat_default_lift_fixmatch_wd_1e_4_pl_entropy_thre_18_budget_6_lr_00003_wd_1e_5/ \
  --cls 810 \