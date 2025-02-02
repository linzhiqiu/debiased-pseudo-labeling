# Best Val Acc 0.39944395422935486 @ epoch 296
# Best Test Acc 0.3539855182170868 @ Best val epoch 296
# Best Test Acc 0.3570230305194855 @ epoch 297
# checkpoint saved in:  checkpoints/supervised/imagenet127_wd_1e_3/11-23-2022-22:10
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_supervised.py \
  /scratch/fercus/imagenet \
  --index_name default \
  --dataset imagenet127 \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --lr 0.003 \
  --weight-decay 1e-3 \
  --cos \
  --epochs 300 \
  --warmup-epoch 10 \
  --dist-url 'tcp://localhost:10101' --multiprocessing-distributed --world-size 1 --rank 0 \
  --self-pretrained pretrained/res50_moco_eman_800ep.pth.tar \
  --amp-opt-level O1 \
  --train-type RandAugment \
  --output checkpoints/supervised/imagenet127_wd_1e_3/ \
  --cls 127 \