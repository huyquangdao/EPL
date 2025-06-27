export CUDA_VISIBLE_DEVICES=3
seed=2

CUDA_VISIBLE_DEVICES=3 accelerate launch --gpu_ids 3 baselines/color/infer_color_policy.py \
     --dataset durecdial \
     --train_data_path data/DuRecDial/data/en_train.txt \
     --dev_data_path data/DuRecDial/data/en_dev.txt \
     --test_data_path data/DuRecDial/data/en_test.txt \
     --bert_dir facebook/bart-base \
     --num_train_epochs 5 \
     --per_device_train_batch_size 1 \
     --per_device_eval_batch_size 1 \
     --gradient_accumulation_steps 1 \
     --num_warmup_steps 3125   \
     --max_sequence_length 512 \
     --max_transition_number 11 \
     --latent_dim 16 \
     --freeze_plm \
     --use_simulated \
     --eval_brownian_bridge \
     --train_use_bridge \
     --infer_use_bridge \
     --max_dec_len 80 \
     --trans_alpha=0.1 \
     --gen_beta=1.0 \
     --kl_gamma=1.0 \
     --learning_rate 2e-5 \
     --max_grad_norm 5 \
     --output_dir ./color_${seed}/ \
     --seed 4

#CUDA_VISIBLE_DEVICES=3 accelerate launch --gpu_ids 3 baselines/tcp/train_tcp.py \
#    --dataset inspired \
#    --train_data_path data/INSPIRED/data/dialog_data/train.tsv \
#    --dev_data_path data/INSPIRED/data/dialog_data/dev.tsv \
#    --test_data_path data/INSPIRED/data/dialog_data/test.tsv \
#    --bert_dir bert-base-cased \
#    --per_device_train_batch_size 8 \
#    --per_device_eval_batch_size 8 \
#    --gradient_accumulation_steps 1 \
#    --num_warmup_steps 512   \
#    --max_sequence_length 512 \
#    --learning_rate 5e-5 \
#    --output_dir ./tcp_inspired_${seed}/ \
#    --seed ${seed}