export CUDA_VISIBLE_DEVICES=5
seed=3

CUDA_VISIBLE_DEVICES=7 accelerate launch --gpu_ids 7 baselines/unimind/train_unimind.py \
   --dataset durecdial \
   --train_data_path data/DuRecDial/data/en_train.txt \
   --dev_data_path data/DuRecDial/data/en_dev.txt \
   --test_data_path data/DuRecDial/data/en_test.txt \
   --tokenizer facebook/bart-base \
   --plm_model facebook/bart-base \
   --num_train_epochs 5 \
   --num_finetune_epochs 2 \
   --do_train \
   --do_finetune \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 1 \
   --num_warmup_steps 3150   \
   --max_sequence_length 512 \
   --max_target_length 100 \
   --learning_rate 5e-5 \
   --output_dir ./unimind/ \
   --seed ${seed}


# CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 baselines/unimind/train_unimind.py \
#     --dataset inspired \
#     --train_data_path data/INSPIRED/data/dialog_data/train.tsv \
#     --dev_data_path data/INSPIRED/data/dialog_data/dev.tsv \
#     --test_data_path data/INSPIRED/data/dialog_data/test.tsv \
#     --tokenizer facebook/bart-base \
#     --plm_model facebook/bart-base \
#     --num_train_epochs 3 \
#     --num_finetune_epochs 3 \
#     --do_train \
#     --do_finetune \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 32 \
#     --gradient_accumulation_steps 1 \
#     --num_warmup_steps 1000   \
#     --max_sequence_length 512 \
#     --max_target_length 100 \
#     --learning_rate 5e-5 \
#     --output_dir ./unimind_inspired/ \
#     --seed ${seed}
