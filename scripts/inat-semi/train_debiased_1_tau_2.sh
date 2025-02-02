# Best Val Acc 0.6471819281578064 @ epoch 528
# Best Test Acc 0.6981052756309509 @ Best val epoch 528
# Best Test Acc 0.6996885538101196 @ epoch 596
# checkpoint saved in:  checkpoints/semi_supervised/inat_debiased_1_tau_2/10-01-2022-22:10
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_semi_supervised.py \
  /scratch/fercus/inat \
  --index_name default \
  --dataset inat \
  --arch FixMatch --backbone resnet50_encoder \
  --debiased True \
  --eman \
  --lr 0.003 \
  --weight-decay 1e-4 \
  --cos \
  --epochs 600 \
  --warmup-epoch 10 \
  --dist-url 'tcp://localhost:11001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --self-pretrained pretrained/res50_moco_eman_800ep.pth.tar \
  --amp-opt-level O1 \
  --threshold 0.98 \
  --tau 0.6 \
  --train-type RandAugment \
  --weak-type DefaultTrain \
  --strong-type RandAugment \
  --multiviews \
  --qhat_m 0.99 \
  --output checkpoints/semi_supervised/inat_debiased_1_tau_2/ \
  --cls 810 \