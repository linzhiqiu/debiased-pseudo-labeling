# Best Val Acc 0.6258476972579956 @ epoch 531
# Best Test Acc 0.679878830909729 @ Best val epoch 531
# Best Test Acc 0.6835481524467468 @ epoch 560
# checkpoint saved in:  checkpoints/semi_supervised/inat_debiased_2_tau_2/09-28-2022-20:31
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
  --threshold 0.90 \
  --tau 0.5 \
  --train-type RandAugment \
  --weak-type DefaultTrain \
  --strong-type RandAugment \
  --multiviews \
  --qhat_m 0.99 \
  --output checkpoints/semi_supervised/inat_debiased_2_tau_2/ \
  --cls 810 \