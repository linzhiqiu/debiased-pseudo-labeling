# Best Val Acc 0.49979549646377563 @ epoch 216
# Best Test Acc 0.4475300908088684 @ Best val epoch 216
# Best Test Acc 0.4475300908088684 @ epoch 216
# checkpoint saved in:  checkpoints/supervised/imagenet127_lift_uniform_wd_1e_5/11-25-2022-23:22
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_supervised.py \
  /ssd0/fercus/imagenet \
  --index_name default_lift_uniform \
  --dataset imagenet127 \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --lr 0.003 \
  --weight-decay 1e-5 \
  --cos \
  --epochs 300 \
  --warmup-epoch 10 \
  --dist-url 'tcp://localhost:10101' --multiprocessing-distributed --world-size 1 --rank 0 \
  --self-pretrained pretrained/res50_moco_eman_800ep.pth.tar \
  --amp-opt-level O1 \
  --train-type RandAugment \
  --output checkpoints/supervised/imagenet127_lift_uniform_wd_1e_5/ \
  --cls 127 \