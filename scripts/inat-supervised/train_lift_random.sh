# Best Val Acc 0.5622162222862244 @ epoch 274
# Best Test Acc 0.6156430840492249 @ Best val epoch 274
# Best Test Acc 0.6201814413070679 @ epoch 284
# checkpoint saved in:  checkpoints/supervised/inat_lift_random/09-14-2022-02:32
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_supervised.py \
  /scratch/fercus/inat \
  --index_name default_lift_random \
  --dataset inat \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --lr 0.003 \
  --weight-decay 1e-4 \
  --cos \
  --epochs 300 \
  --warmup-epoch 10 \
  --dist-url 'tcp://localhost:12101' --multiprocessing-distributed --world-size 1 --rank 0 \
  --self-pretrained pretrained/res50_moco_eman_800ep.pth.tar \
  --amp-opt-level O1 \
  --train-type RandAugment \
  --output checkpoints/supervised/inat_lift_random/ \
  --cls 810 \