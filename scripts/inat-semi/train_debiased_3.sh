# Best Val Acc 0.6490240693092346 @ epoch 592
# Best Test Acc 0.6879002451896667 @ Best val epoch 592
# Best Test Acc 0.6896288394927979 @ epoch 494
# checkpoint saved in:  checkpoints/semi_supervised/inat_debiased_3/09-08-2022-21:41
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_semi_supervised.py \
  /scratch/fercus/semi_inat \
  --index_name default \
  --dataset semi_inat \
  --arch FixMatch --backbone resnet50_encoder \
  --debiased True \
  --eman \
  --lr 0.003 \
  --weight-decay 1e-3 \
  --cos \
  --epochs 600 \
  --warmup-epoch 10 \
  --dist-url 'tcp://localhost:11301' --multiprocessing-distributed --world-size 1 --rank 0 \
  --self-pretrained pretrained/res50_moco_eman_800ep.pth.tar \
  --amp-opt-level O1 \
  --threshold 0.8 \
  --tau 0.4 \
  --train-type RandAugment \
  --weak-type DefaultTrain \
  --strong-type RandAugment \
  --multiviews \
  --qhat_m 0.99 \
  --output checkpoints/semi_supervised/inat_debiased_3/ \
  --cls 810 \