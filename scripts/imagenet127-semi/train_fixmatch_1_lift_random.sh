# Best Val Acc 0.6357473134994507 @ epoch 12
# Best Test Acc 0.5853362083435059 @ Best val epoch 12
# Best Test Acc 0.586764395236969 @ epoch 13
# checkpoint saved in:  checkpoints/semi_supervised/imagenet127_fixmatch_1_lift_random/09-27-2022-21:06
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_semi_supervised.py \
  /scratch/fercus/imagenet \
  --index_name default_lift_random \
  --dataset imagenet127 \
  --arch FixMatch --backbone resnet50_encoder \
  --debiased False \
  --eman \
  --lr 0.003 \
  --weight-decay 1e-4 \
  --cos \
  --epochs 50 \
  --warmup-epoch 5 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --self-pretrained pretrained/res50_moco_eman_800ep.pth.tar \
  --amp-opt-level O1 \
  --threshold 0.95 \
  --train-type RandAugment \
  --weak-type DefaultTrain \
  --strong-type RandAugment \
  --multiviews \
  --output checkpoints/semi_supervised/imagenet127_fixmatch_1_lift_random/ \
  --cls 127 \