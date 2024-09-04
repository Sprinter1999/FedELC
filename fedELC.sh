# -- store logs-- #
mkdir fedELC_logs

pip install scikit-learn

python main_fed_LNL.py \
--dataset cifar10 \
--model resnet18 \
--epochs 120 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition dirichlet \
--dd_alpha 1.0 \
--epoch_of_stage1 20 \
--lamda_pencil 1000 \
--method fedELC | tee ./fedELC_logs/symm04_dirichlet10.txt

# python main_fed_LNL.py \
# --dataset cifar10 \
# --model resnet18 \
# --epochs 120 \
# --noise_type_lst pairflip \
# --noise_group_num 100  \
# --group_noise_rate 0.0 0.4 \
# --partition dirichlet \
# --dd_alpha 1.0 \
# --epoch_of_stage1 20 \
# --lamda_pencil 1000 \
# --method fedELC | tee ./fedELC_logs/pair04_dirichlet10.txt

# python main_fed_LNL.py \
# --dataset cifar10 \
# --model resnet18 \
# --epochs 120 \
# --noise_type_lst symmetric pairflip \
# --noise_group_num 50 50 \
# --group_noise_rate 0.0 0.4 0.0 0.4 \
# --partition dirichlet \
# --dd_alpha 1.0 \
# --method fedELC | tee ./fedELC_logs/mixed04_dirichlet10.txt
