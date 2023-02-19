# Best Val Acc 0.5348218083381653 @ epoch 298
# Best Test Acc 0.5892146825790405 @ Best val epoch 298
# Best Test Acc 0.5892146825790405 @ epoch 298
# checkpoint saved in:  checkpoints/supervised/inat_lift_uniform_wd_1e_3/11-24-2022-01:39
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_supervised.py \
  /scratch/fercus/inat \
  --index_name default_lift_uniform \
  --dataset inat \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --lr 0.003 \
  --weight-decay 1e-3 \
  --cos \
  --epochs 300 \
  --warmup-epoch 10 \
  --dist-url 'tcp://localhost:11101' --multiprocessing-distributed --world-size 1 --rank 0 \
  --self-pretrained pretrained/res50_moco_eman_800ep.pth.tar \
  --amp-opt-level O1 \
  --train-type RandAugment \
  --output checkpoints/supervised/inat_lift_uniform_wd_1e_3/ \
  --cls 810 \