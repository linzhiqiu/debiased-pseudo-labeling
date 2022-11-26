# Best Val Acc 0.5516864657402039 @ epoch 296
# Best Test Acc 0.6101431846618652 @ Best val epoch 296
# Best Test Acc 0.6144336462020874 @ epoch 299
# checkpoint saved in:  checkpoints/supervised/inat_lift_tail_uniform_1e_3/11-24-2022-12:18
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_supervised.py \
  /scratch/fercus/inat \
  --index_name default_lift_tail_uniform \
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
  --output checkpoints/supervised/inat_lift_tail_uniform_1e_3/ \
  --cls 810 \