# Best Val Acc 0.4110061824321747 @ epoch 261
# Best Test Acc 0.3684348464012146 @ Best val epoch 261
# Best Test Acc 0.3697604537010193 @ epoch 289
# checkpoint saved in:  checkpoints/supervised/imagenet127_lift_uniform_wd_1e_3/11-29-2022-01:38
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_supervised.py \
  /scratch/fercus/imagenet \
  --index_name default_lift_uniform \
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
  --output checkpoints/supervised/imagenet127_lift_uniform_wd_1e_3/ \
  --cls 127 \