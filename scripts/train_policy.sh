export CUDA_VISIBLE_DEVICES=4
seed=22

CUDA_VISIBLE_DEVICES=4 accelerate launch --gpu_ids 4 train_policy.py \
    --dataset durecdial \
    --train_data_path data/DuRecDial/data/en_train.txt \
    --dev_data_path data/DuRecDial/data/en_dev.txt \
    --test_data_path data/DuRecDial/data/en_test.txt \
    --tokenizer bert-base-cased \
    --plm_model bert-base-cased \
    --num_train_epochs 5 \
    --hidden_size 128 \
    --lm_size 768 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 6349   \
    --max_sequence_length 512 \
    --learning_rate 5e-5 \
    --output_dir ./policy_model_bert_${seed}/ \
    --seed ${seed}

# CUDA_VISIBLE_DEVICES=4 accelerate launch --gpu_ids 4 train_policy.py \
#     --dataset inspired \
#     --train_data_path data/INSPIRED/data/dialog_data/train.tsv \
#     --dev_data_path data/INSPIRED/data/dialog_data/dev.tsv \
#     --test_data_path data/INSPIRED/data/dialog_data/test.tsv \
#     --tokenizer bert-base-cased \
#     --plm_model bert-base-cased \
#     --num_train_epochs 10 \
#     --hidden_size 128 \
#     --lm_size 768 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 32 \
#     --gradient_accumulation_steps 1 \
#     --num_warmup_steps 1000   \
#     --max_sequence_length 512 \
#     --learning_rate 1e-5 \
#     --output_dir ./policy_model_bert_${seed}_inspired/ \
#     --seed ${seed}