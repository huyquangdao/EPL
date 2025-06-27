export CUDA_VISIBLE_DEVICES=3

CUDA_VISIBLE_DEVICES=3 accelerate launch --gpu_ids 3 baselines/tcp/infer_tcp_policy.py \
    --dataset durecdial \
    --train_data_path data/DuRecDial/data/en_train.txt \
    --dev_data_path data/DuRecDial/data/en_dev.txt \
    --test_data_path data/DuRecDial/data/en_test.txt \
    --bert_dir bert-base-cased \
    --per_device_eval_batch_size 8 \
    --max_sequence_length 512 \
    --output_dir ./tcp_1/ \
    --seed 21

# CUDA_VISIBLE_DEVICES=3 accelerate launch --gpu_ids 3 baselines/tcp/train_tcp.py \
#     --dataset inspired \
#     --train_data_path data/INSPIRED/data/dialog_data/train.tsv \
#     --dev_data_path data/INSPIRED/data/dialog_data/dev.tsv \
#     --test_data_path data/INSPIRED/data/dialog_data/test.tsv \
#     --bert_dir bert-base-cased \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1 \
#     --num_warmup_steps 512   \
#     --max_sequence_length 512 \
#     --learning_rate 5e-5 \
#     --output_dir ./tcp/ \
#     --seed 21