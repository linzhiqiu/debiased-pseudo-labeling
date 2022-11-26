# Best Val Acc 0.5696818828582764 @ epoch 266
# Best Test Acc 0.6277998685836792 @ Best val epoch 266
# Best Test Acc 0.6327565312385559 @ epoch 294
# checkpoint saved in:  checkpoints/supervised/inat_lift_tail_uniform_1e_5/11-24-2022-12:19
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_supervised.py \
  /ssd0/fercus/inat \
  --index_name default_lift_tail_uniform \
  --dataset inat \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --lr 0.003 \
  --weight-decay 1e-5 \
  --cos \
  --epochs 300 \
  --warmup-epoch 10 \
  --dist-url 'tcp://localhost:11101' --multiprocessing-distributed --world-size 1 --rank 0 \
  --self-pretrained pretrained/res50_moco_eman_800ep.pth.tar \
  --amp-opt-level O1 \
  --train-type RandAugment \
  --output checkpoints/supervised/inat_lift_tail_uniform_1e_5/ \
  --cls 810 \