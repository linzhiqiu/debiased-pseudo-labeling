# Best Val Acc 0.6694944500923157 @ epoch 491
# Best Test Acc 0.7342702746391296 @ Best val epoch 491
# Best Test Acc 0.7374085783958435 @ epoch 561
# checkpoint saved in:  checkpoints/semi_supervised/inat_fixmatch_lift_random/09-21-2022-01:11
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_semi_supervised.py \
  /scratch/fercus/inat \
  --index_name default_lift_random \
  --dataset inat \
  --arch FixMatch --backbone resnet50_encoder \
  --debiased False \
  --eman \
  --lr 0.003 \
  --weight-decay 1e-4 \
  --cos \
  --epochs 600 \
  --warmup-epoch 10 \
  --dist-url 'tcp://localhost:11001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --self-pretrained pretrained/res50_moco_eman_800ep.pth.tar \
  --amp-opt-level O1 \
  --threshold 0.95 \
  --train-type RandAugment \
  --weak-type DefaultTrain \
  --strong-type RandAugment \
  --multiviews \
  --output checkpoints/semi_supervised/inat_fixmatch_lift_random/ \
  --cls 810 \