# Best Val Acc 0.6193982362747192 @ epoch 265
# Best Test Acc 0.6676532626152039 @ Best val epoch 265
# Best Test Acc 0.6708632707595825 @ epoch 259
# checkpoint not available

# Best Val Acc 0.613990068435669 @ epoch 265
# Best Test Acc 0.661480724811554 @ Best val epoch 265
# Best Test Acc 0.6701223850250244 @ epoch 295
# checkpoint saved in:  checkpoints/supervised/semi_inat_lift_random/09-03-2022-01:09
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_supervised.py \
  /scratch/fercus/semi_inat \
  --index_name default_lift_random \
  --dataset semi_inat \
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
  --output checkpoints/supervised/semi_inat_lift_random/ \
  --cls 810 \