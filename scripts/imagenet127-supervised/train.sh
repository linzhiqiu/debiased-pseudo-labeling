# Best Val Acc 0.48302581906318665 @ epoch 177
# Best Test Acc 0.4162408411502838 @ Best val epoch 177
# Best Test Acc 0.42693695425987244 @ epoch 265
# checkpoint saved in:  checkpoints/supervised/imagenet127/09-16-2022-11:48
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_supervised.py \
  /scratch/fercus/imagenet \
  --index_name default \
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
  --output checkpoints/supervised/imagenet127/ \
  --cls 127 \