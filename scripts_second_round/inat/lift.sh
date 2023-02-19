python lifting.py --dataset inat --index_name default --threshold 18 --budget 6 --lift pl --model_name wd_1e_5 --algo supervised
python lifting.py --dataset inat --index_name default --threshold 18 --budget 6 --lift pl_entropy --model_name wd_1e_5 --algo supervised
python lifting.py --dataset inat --index_name default --threshold 18 --budget 6 --lift pl --model_name wd_1e_4 --algo fixmatch
python lifting.py --dataset inat --index_name default --threshold 18 --budget 6 --lift pl_entropy --model_name wd_1e_4 --algo fixmatch
python lifting.py --dataset inat --index_name default --threshold 18 --budget 6 --lift pl --model_name wd_1e_4 --algo debiased
python lifting.py --dataset inat --index_name default --threshold 18 --budget 6 --lift pl_entropy --model_name wd_1e_4 --algo debiased