# Best Val Acc 0.5231874585151672 @ epoch 293
# Best Test Acc 0.5781357288360596 @ Best val epoch 293
# Best Test Acc 0.5783609747886658 @ epoch 287
# checkpoint saved in:  checkpoints/supervised/inat_wd_1e_5/11-23-2022-17:26
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_supervised.py \
  /ssd0/fercus/inat \
  --index_name default \
  --dataset inat \
  --arch FixMatch --backbone resnet50_encoder \
  --eman \
  --lr 0.003 \
  --weight-decay 1e-5 \
  --cos \
  --epochs 300 \
  --warmup-epoch 10 \
  --dist-url 'tcp://localhost:11001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --self-pretrained pretrained/res50_moco_eman_800ep.pth.tar \
  --amp-opt-level O1 \
  --train-type RandAugment \
  --output checkpoints/supervised/inat_wd_1e_5/ \
  --cls 810 \