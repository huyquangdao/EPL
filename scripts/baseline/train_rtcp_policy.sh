export CUDA_VISIBLE_DEVICES=6
seed=2

#CUDA_VISIBLE_DEVICES=6 accelerate launch --gpu_ids 6 baselines/rtcp/train_policy_rtcp.py \
#    --train_data_path data/DuRecDial/data/en_train.txt \
#    --dev_data_path data/DuRecDial/data/en_dev.txt \
#    --test_data_path data/DuRecDial/data/en_test.txt \
#    --tokenizer bert-base-cased \
#    --plm_model bert-base-cased \
#    --num_train_epochs 5 \
#    --lm_size 768 \
#    --ffn_size 3072 \
#    --n_layers 12 \
#    --n_heads 8 \
#    --fc_size 128 \
#    --per_device_train_batch_size 8 \
#    --per_device_eval_batch_size 8 \
#    --gradient_accumulation_steps 1 \
#    --num_warmup_steps 6345   \
#    --max_sequence_length 512 \
#    --learning_rate 5e-5 \
#    --output_dir ./rtcp/ \
#    --seed 21

CUDA_VISIBLE_DEVICES=6 accelerate launch --gpu_ids 6 baselines/rtcp/train_policy_rtcp.py \
    --dataset inspired \
    --train_data_path data/INSPIRED/data/dialog_data/train.tsv \
    --dev_data_path data/INSPIRED/data/dialog_data/dev.tsv \
    --test_data_path data/INSPIRED/data/dialog_data/test.tsv \
    --tokenizer bert-base-cased \
    --plm_model bert-base-cased \
    --num_train_epochs 5 \
    --lm_size 768 \
    --ffn_size 3072 \
    --n_layers 12 \
    --n_heads 8 \
    --fc_size 128 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 1000   \
    --max_sequence_length 512 \
    --learning_rate 5e-5 \
    --output_dir ./rtcp_inspired_retrained_${seed}/ \
    --seed ${seed}