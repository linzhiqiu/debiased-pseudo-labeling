# Best Val Acc 0.6544714570045471 @ epoch 214
# Best Test Acc 0.7084341645240784 @ Best val epoch 214
# Best Test Acc 0.7094455361366272 @ epoch 20
# checkpoint saved in:  checkpoints/supervised/inat_default_lift_debiased_wd_1e_4_pl_thre_18_budget_6_lr_00003_wd_1e_5/12-02-2022-00:57
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_supervised.py \
  /scratch/fercus/inat \
  --index_name default_lift_debiased_wd_1e_4_pl_thre_18_budget_6 \
  --dataset inat \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --lr 0.0003 \
  --weight-decay 1e-5 \
  --cos \
  --epochs 300 \
  --warmup-epoch 10 \
  --dist-url 'tcp://localhost:11001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained checkpoints/semi_supervised/inat_debiased_tau_4/10-09-2022-13:24/best_val_model.pth.tar \
  --amp-opt-level O1 \
  --train-type RandAugment \
  --output checkpoints/supervised/inat_default_lift_debiased_wd_1e_4_pl_thre_18_budget_6_lr_00003_wd_1e_5/ \
  --cls 810 \