# Best Val Acc 0.669148325920105 @ epoch 530
# Best Test Acc 0.7259998321533203 @ Best val epoch 530
# Best Test Acc 0.7288997769355774 @ epoch 562
# checkpoint saved in:  checkpoints/semi_supervised/inat_debiased_tau_4_lift_uniform/10-12-2022-14:38
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_semi_supervised.py \
  /scratch/fercus/inat \
  --index_name default_lift_uniform \
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
  --threshold 0.95 \
  --tau 0.9 \
  --train-type RandAugment \
  --weak-type DefaultTrain \
  --strong-type RandAugment \
  --multiviews \
  --qhat_m 0.99 \
  --output checkpoints/semi_supervised/inat_debiased_tau_4_lift_uniform/ \
  --cls 810 \