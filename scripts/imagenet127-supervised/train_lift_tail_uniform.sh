# Best Val Acc 0.6656914353370667 @ epoch 599
# Best Test Acc 0.7305269837379456 @ Best val epoch 599
# Best Test Acc 0.7310342788696289 @ epoch 565
# checkpoint saved in:  checkpoints/semi_supervised/inat_fixmatch_lift_tail_uniform/10-02-2022-11:58
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