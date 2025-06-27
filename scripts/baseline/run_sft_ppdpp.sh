export CUDA_VISIBLE_DEVICES=6
seed=1

#  CUDA_VISIBLE_DEVICES=1 accelerate launch --gpu_ids 1 baselines/ppdpp/sft.py \
#      --dataset durecdial \
#      --train_data_path data/DuRecDial/data/en_train.txt \
#      --dev_data_path data/DuRecDial/data/en_dev.txt \
#      --test_data_path data/DuRecDial/data/en_test.txt \
#      --tokenizer roberta-large \
#      --plm_model roberta-large \
#      --num_train_epochs 5 \
#      --hidden_size 128 \
#      --lm_size 768 \
#      --per_device_train_batch_size 16 \
#      --per_device_eval_batch_size 32 \
#      --gradient_accumulation_steps 1 \
#      --num_warmup_steps 6349   \
#      --max_sequence_length 512 \
#      --learning_rate 5e-5 \
#      --output_dir ./ppdpp_${seed}/ \
#      --seed ${seed}

CUDA_VISIBLE_DEVICES=6 accelerate launch --gpu_ids 6 baselines/ppdpp/sft.py \
    --dataset inspired \
    --train_data_path data/INSPIRED/data/dialog_data/train.tsv \
    --dev_data_path data/INSPIRED/data/dialog_data/dev.tsv \
    --test_data_path data/INSPIRED/data/dialog_data/test.tsv \
    --tokenizer roberta-large \
    --plm_model roberta-large \
    --num_train_epochs 5 \
    --hidden_size 128 \
    --lm_size 768 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 1024   \
    --max_sequence_length 512 \
    --learning_rate 5e-5 \
    --output_dir ./ppdpp_${seed}_inspired/ \
    --seed ${seed}