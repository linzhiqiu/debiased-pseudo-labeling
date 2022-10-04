# Best Val Acc 0.5513914227485657 @ epoch 270
# Best Test Acc 0.49057456851005554 @ Best val epoch 270
# Best Test Acc 0.4950324594974518 @ epoch 247
# checkpoint saved in:  checkpoints/supervised/imagenet127_lift_tail_uniform/10-03-2022-12:16
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_supervised.py \
  /scratch/fercus/imagenet \
  --index_name default_lift_tail_uniform \
  --dataset imagenet127 \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --lr 0.003 \
  --weight-decay 1e-4 \
  --cos \
  --epochs 300 \
  --warmup-epoch 10 \
  --dist-url 'tcp://localhost:10101' --multiprocessing-distributed --world-size 1 --rank 0 \
  --self-pretrained pretrained/res50_moco_eman_800ep.pth.tar \
  --amp-opt-level O1 \
  --train-type RandAugment \
  --output checkpoints/supervised/imagenet127_lift_tail_uniform/ \
  --cls 127 \