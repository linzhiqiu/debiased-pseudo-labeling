# Best Val Acc 0.642880380153656 @ epoch 1
# Best Test Acc 0.703308641910553 @ Best val epoch 1
# Best Test Acc 0.7038753032684326 @ epoch 0
# checkpoint saved in:  checkpoints/supervised/inat_default_from_fixmatch_lr_0003_wd_1e_5/11-29-2022-13:02
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_supervised.py \
  /scratch/fercus/inat \
  --index_name default \
  --dataset inat \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --lr 0.003 \
  --weight-decay 1e-5 \
  --cos \
  --epochs 300 \
  --warmup-epoch 10 \
  --dist-url 'tcp://localhost:11001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained checkpoints/semi_supervised/inat_fixmatch/09-16-2022-11:30/best_val_model.pth.tar \
  --amp-opt-level O1 \
  --train-type RandAugment \
  --output checkpoints/supervised/inat_default_from_fixmatch_lr_0003_wd_1e_5/ \
  --cls 810 \