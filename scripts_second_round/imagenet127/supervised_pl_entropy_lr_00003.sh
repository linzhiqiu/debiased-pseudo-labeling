# Best Val Acc 0.5256801247596741 @ epoch 18
# Best Test Acc 0.4573608338832855 @ Best val epoch 18
# Best Test Acc 0.4630727469921112 @ epoch 42
# checkpoint saved in:  checkpoints/supervised/imagenet127_wd_1e_5_default_lift_supervised_wd_1e_5_pl_entropy_thre_25_budget_12_lr_00003_wd_1e_5/11-29-2022-01:39
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_supervised.py \
  /scratch/fercus/imagenet \
  --index_name default_lift_supervised_wd_1e_5_pl_entropy_thre_25_budget_12 \
  --dataset imagenet127 \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --lr 0.0003 \
  --weight-decay 1e-5 \
  --cos \
  --epochs 300 \
  --warmup-epoch 10 \
  --dist-url 'tcp://localhost:10111' --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained checkpoints/supervised/imagenet127_wd_1e_5/11-24-2022-17:36/best_val_model.pth.tar \
  --amp-opt-level O1 \
  --train-type RandAugment \
  --output checkpoints/supervised/imagenet127_wd_1e_5_default_lift_supervised_wd_1e_5_pl_entropy_thre_25_budget_12_lr_00003_wd_1e_5/ \
  --cls 127 \