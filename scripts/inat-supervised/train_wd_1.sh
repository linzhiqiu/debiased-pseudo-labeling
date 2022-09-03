# Best Val Acc 0.5929434895515442 @ epoch 295
# Best Test Acc 0.649134635925293 @ Best val epoch 295
# Best Test Acc 0.6520977020263672 @ epoch 293
# checkpoint saved in:  checkpoints/supervised/semi_inat_wd_1/09-01-2022-02:29

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_supervised.py \
  /scratch/fercus/semi_inat \
  --index_name default \
  --dataset semi_inat \
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
  --output checkpoints/supervised/semi_inat_wd_1/ \
  --cls 810 \