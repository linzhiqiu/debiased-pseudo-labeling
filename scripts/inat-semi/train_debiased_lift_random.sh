# Best Val Acc 0.6232173442840576 @ epoch 242
# Best Test Acc 0.660987138748169 @ Best val epoch 242
# Best Test Acc 0.6612338423728943 @ epoch 236
# checkpoint saved in:  checkpoints/semi_supervised/inat_debiased_lift_random/08-31-2022-13:41

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_semi_supervised.py \
  /scratch/fercus/semi_inat \
  --index_name default_lift_random \
  --dataset semi_inat \
  --arch FixMatch --backbone resnet50_encoder \
  --debiased True \
  --eman \
  --lr 0.003 \
  --weight-decay 1e-4 \
  --cos \
  --epochs 250 \
  --warmup-epoch 25 \
  --dist-url 'tcp://localhost:11001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --self-pretrained pretrained/res50_moco_eman_800ep.pth.tar \
  --amp-opt-level O1 \
  --threshold 0.8 \
  --tau 0.4 \
  --train-type RandAugment \
  --weak-type DefaultTrain \
  --strong-type RandAugment \
  --multiviews \
  --qhat_m 0.99 \
  --output checkpoints/semi_supervised/inat_debiased_lift_random/ \
  --cls 810 \