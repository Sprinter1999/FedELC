mkdir eval_logs

pip install scikit-learn

python main_fed_LNL.py \
--dataset cifar10 \
--model resnet18 \
--epochs 120 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50 \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partition dirichlet \
--dd_alpha 1.0 \
--method fedELC | tee ./eval_logs/fedelc_mixed04_dirichlet10.txt