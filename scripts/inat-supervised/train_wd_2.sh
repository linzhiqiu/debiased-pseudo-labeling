# Best Val Acc 0.6100383400917053 @ epoch 283
# Best Test Acc 0.6629619002342224 @ Best val epoch 283
# Best Test Acc 0.6696288585662842 @ epoch 286
# checkpoint saved in:  checkpoints/supervised/semi_inat_wd_2/09-01-2022-12:13

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_supervised.py \
  /scratch/fercus/semi_inat \
  --index_name default \
  --dataset semi_inat \
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
  --output checkpoints/supervised/semi_inat_wd_2/ \
  --cls 810 \