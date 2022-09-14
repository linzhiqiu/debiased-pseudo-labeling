# Best Val Acc 0.595188319683075 @ epoch 592
# Best Test Acc 0.6528382897377014 @ Best val epoch 592
# Best Test Acc 0.6548136472702026 @ epoch 598
# checkpoint saved in:  checkpoints/supervised/semi_inat_1/09-10-2022-16:01
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_supervised.py \
  /scratch/fercus/semi_inat \
  --index_name default \
  --dataset semi_inat \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --lr 0.03 \
  --weight-decay 1e-4 \
  --cos \
  --epochs 600 \
  --warmup-epoch 10 \
  --dist-url 'tcp://localhost:11101' --multiprocessing-distributed --world-size 1 --rank 0 \
  --self-pretrained pretrained/res50_moco_eman_800ep.pth.tar \
  --amp-opt-level O1 \
  --train-type RandAugment \
  --output checkpoints/supervised/semi_inat_1/ \
  --cls 810 \