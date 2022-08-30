CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_labeled_only.py \
  --arch ResNet --backbone resnet50_encoder \
  --norm BN \ # or SyncBN?
  # TO FIX
  --eman \
  --lr 0.003 \
  --cos \
  --epochs 50 \
  --warmup-epoch 5 \
  --trainindex_x train_0.2p_index.csv --trainindex_u train_99.8p_index.csv \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --self-pretrained pretrained/res50_moco_eman_800ep.pth.tar \
  --amp-opt-level O1 \
  --threshold 0.7 \
  --tau 0.4 \
  --CLDLoss \
  --lambda-cld 0.1 \
  --multiviews \
  --qhat_m 0.99 \
  --use_clip \
  --output checkpoints/imagenet127_supervised/ \ # Modify this with hyperparameter
  --cls 1000 \
  imagenet/