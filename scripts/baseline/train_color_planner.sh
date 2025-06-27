export CUDA_VISIBLE_DEVICES=7
seed=3

CUDA_VISIBLE_DEVICES=2 accelerate launch --gpu_ids 2 baselines/color/train_planner.py \
     --dataset durecdial \
     --train_data_path data/DuRecDial/data/en_train.txt \
     --dev_data_path data/DuRecDial/data/en_dev.txt \
     --test_data_path data/DuRecDial/data/en_test.txt \
     --bert_dir facebook/bart-base \
     --num_train_epochs 5 \
     --per_device_train_batch_size 16 \
     --per_device_eval_batch_size 16 \
     --gradient_accumulation_steps 1 \
     --num_warmup_steps 3125   \
     --max_sequence_length 512 \
     --max_transition_number 11 \
     --latent_dim 16 \
     --freeze_plm \
     --use_simulated \
     --eval_brownian_bridge \
     --train_use_bridge \
     --trans_alpha=0.1 \
     --gen_beta=1.0 \
     --kl_gamma=1.0 \
     --learning_rate 2e-5 \
     --max_grad_norm 5 \
     --output_dir ./color_${seed}/ \
     --load_checkpoint_bridge \
     --seed 4

# CUDA_VISIBLE_DEVICES=7 accelerate launch --gpu_ids 7 baselines/color/train_planner.py \
#     --dataset inspired \
#     --train_data_path data/INSPIRED/data/dialog_data/train.tsv \
#     --dev_data_path data/INSPIRED/data/dialog_data/dev.tsv \
#     --test_data_path data/INSPIRED/data/dialog_data/test.tsv \
#      --bert_dir facebook/bart-base \
#      --num_train_epochs 5 \
#      --per_device_train_batch_size 16 \
#      --per_device_eval_batch_size 16 \
#      --gradient_accumulation_steps 1 \
#      --num_warmup_steps 512   \
#      --max_sequence_length 1000 \
#      --max_transition_number 36 \
#      --latent_dim 16 \
#      --freeze_plm \
#      --use_simulated \
#      --eval_brownian_bridge \
#      --train_use_bridge \
#      --trans_alpha 0.1 \
#      --gen_beta 1.0 \
#      --kl_gamma 1.0 \
#      --learning_rate 5e-5 \
#      --max_grad_norm 5 \
#      --output_dir ./color_inspired_${seed}/ \
#      --load_checkpoint_bridge \
#      --seed 4