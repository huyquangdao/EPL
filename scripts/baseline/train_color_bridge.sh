export CUDA_VISIBLE_DEVICES=2
seed=3

CUDA_VISIBLE_DEVICES=2 accelerate launch --gpu_ids 2 baselines/color/train_bridge.py \
     --dataset durecdial \
     --train_data_path data/DuRecDial/data/en_train.txt \
     --dev_data_path data/DuRecDial/data/en_dev.txt \
     --test_data_path data/DuRecDial/data/en_test.txt \
     --bert_dir facebook/bart-base \
     --num_train_epochs 10 \
     --per_device_train_batch_size 64 \
     --per_device_eval_batch_size 1 \
     --gradient_accumulation_steps 1 \
     --num_warmup_steps 3125 \
     --use_simulated  \
     --max_sequence_length 512 \
     --max_transition_number 11 \
     --latent_dim 16 \
     --freeze_plm \
     --eval_brownian_bridge \
     --learning_rate 2e-4 \
     --max_grad_norm 5 \
     --output_dir ./color_${seed}/ \
     --seed ${seed}

# CUDA_VISIBLE_DEVICES=2 accelerate launch --gpu_ids 2 baselines/color/train_bridge.py \
#     --dataset inspired \
#     --train_data_path data/INSPIRED/data/dialog_data/train.tsv \
#     --dev_data_path data/INSPIRED/data/dialog_data/dev.tsv \
#     --test_data_path data/INSPIRED/data/dialog_data/test.tsv \
#      --bert_dir facebook/bart-base \
#      --num_train_epochs 5 \
#      --per_device_train_batch_size 64 \
#      --per_device_eval_batch_size 1 \
#      --gradient_accumulation_steps 1 \
#      --num_warmup_steps 1000 \
#      --use_simulated  \
#      --max_sequence_length 512 \
#      --max_transition_number 36 \
#      --latent_dim 16 \
#      --freeze_plm \
#      --eval_brownian_bridge \
#      --learning_rate 1e-4 \
#      --max_grad_norm 5 \
#      --output_dir ./color_inspired_${seed}/ \
#      --seed ${seed}