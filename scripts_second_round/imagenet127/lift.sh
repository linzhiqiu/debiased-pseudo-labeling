python lifting.py --dataset imagenet127 --index_name default --threshold 25 --budget 12 --lift pl --model_name wd_1e_5 --algo supervised
python lifting.py --dataset imagenet127 --index_name default --threshold 25 --budget 12 --lift pl_entropy --model_name wd_1e_5 --algo supervised
CUDA_VISIBLE_DEVICES=2 python lifting.py --dataset imagenet127 --index_name default --threshold 25 --budget 12 --lift pl --model_name wd_1e_4 --algo fixmatch
CUDA_VISIBLE_DEVICES=3 python lifting.py --dataset imagenet127 --index_name default --threshold 25 --budget 12 --lift pl_entropy --model_name wd_1e_4 --algo fixmatch
CUDA_VISIBLE_DEVICES=4 python lifting.py --dataset imagenet127 --index_name default --threshold 25 --budget 12 --lift pl --model_name wd_1e_4 --algo debiased
CUDA_VISIBLE_DEVICES=5 python lifting.py --dataset imagenet127 --index_name default --threshold 25 --budget 12 --lift pl_entropy --model_name wd_1e_4 --algo debiased