# Best Val Acc 0.6247315406799316 @ epoch 298
# Best Test Acc 0.6898749470710754 @ Best val epoch 298
# Best Test Acc 0.6943196058273315 @ epoch 299
# Checkpoint not available

# Best Val Acc 0.2836780548095703 @ epoch 285
# Best Test Acc 0.3056783080101013 @ Best val epoch 285
# Best Test Acc 0.30814749002456665 @ epoch 291
# checkpoint saved in:  checkpoints/supervised/semi_inat_lift_tail/09-07-2022-21:52
# on autobot?
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_supervised.py \
  /scratch/fercus/semi_inat \
  --index_name default_lift_tail \
  --dataset semi_inat \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --lr 0.003 \
  --weight-decay 1e-4 \
  --cos \
  --epochs 300 \
  --warmup-epoch 10 \
  --dist-url 'tcp://localhost:11101' --multiprocessing-distributed --world-size 1 --rank 0 \
  --self-pretrained pretrained/res50_moco_eman_800ep.pth.tar \
  --amp-opt-level O1 \
  --train-type RandAugment \
  --output checkpoints/supervised/semi_inat_lift_tail/ \
  --cls 810 \