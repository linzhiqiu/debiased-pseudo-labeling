# Best Val Acc 0.5044260025024414 @ epoch 213
# Best Test Acc 0.4353228211402893 @ Best val epoch 213
# Best Test Acc 0.4384964406490326 @ epoch 211
# checkpoint saved in:  checkpoints/supervised/imagenet127_wd_1e_5/11-24-2022-17:36
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_supervised.py \
  /scratch/fercus/imagenet \
  --index_name default \
  --dataset imagenet127 \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --lr 0.003 \
  --weight-decay 1e-5 \
  --cos \
  --epochs 300 \
  --warmup-epoch 10 \
  --dist-url 'tcp://localhost:10111' --multiprocessing-distributed --world-size 1 --rank 0 \
  --self-pretrained pretrained/res50_moco_eman_800ep.pth.tar \
  --amp-opt-level O1 \
  --train-type RandAugment \
  --output checkpoints/supervised/imagenet127_wd_1e_5/ \
  --cls 127 \