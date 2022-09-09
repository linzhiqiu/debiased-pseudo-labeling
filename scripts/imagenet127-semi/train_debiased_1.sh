# Best Val Acc 0.732079803943634 @ epoch 46
# Best Test Acc 0.7095403075218201 @ Best val epoch 46
# Best Test Acc 0.7153311967849731 @ epoch 48
# checkpoint saved in:  checkpoints/semi_supervised/imagenet127_debiased_1/09-02-2022-19:14
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_semi_supervised.py \
  /scratch/fercus/imagenet \
  --index_name default \
  --dataset imagenet127 \
  --arch FixMatch --backbone resnet50_encoder \
  --debiased True \
  --eman \
  --lr 0.03 \
  --weight-decay 1e-4 \
  --cos \
  --epochs 50 \
  --warmup-epoch 5 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --self-pretrained pretrained/res50_moco_eman_800ep.pth.tar \
  --amp-opt-level O1 \
  --threshold 0.8 \
  --tau 0.4 \
  --train-type RandAugment \
  --weak-type DefaultTrain \
  --strong-type RandAugment \
  --multiviews \
  --qhat_m 0.99 \
  --output checkpoints/semi_supervised/imagenet127_debiased_1/ \
  --cls 127 \