CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_supervised.py \
  /ssd0/fercus/inat \
  --index_name default_lift_supervised_wd_1e_5_pl_entropy_all_fp_thre_18_budget_6 \
  --dataset inat \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --lr 0.0003 \
  --weight-decay 1e-5 \
  --cos \
  --epochs 300 \
  --warmup-epoch 10 \
  --dist-url 'tcp://localhost:11101' --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained checkpoints/supervised/inat_wd_1e_5/11-23-2022-17:26/best_val_model.pth.tar \
  --amp-opt-level O1 \
  --train-type RandAugment \
  --output checkpoints/supervised/inat_default_lift_supervised_wd_1e_5_pl_entropy_all_fp_thre_18_budget_6_lr_00003_wd_1e_5/ \
  --cls 810 \