export CUDA_VISIBLE_DEVICES=1

## infer goal
#CUDA_VISIBLE_DEVICES=1 accelerate launch --gpu_ids 1 baselines/unimind/infer_goal_unimind.py \
#    --train_data_path data/DuRecDial/data/en_train.txt \
#    --dev_data_path data/DuRecDial/data/en_dev.txt \
#    --test_data_path data/DuRecDial/data/en_test.txt \
#    --tokenizer facebook/bart-base \
#    --plm_model facebook/bart-base \
#    --per_device_eval_batch_size 32 \
#    --max_sequence_length 512 \
#    --max_target_length 80 \
#    --output_dir ./unimind/goal/ \
#    --seed 12
#
## #infer topic
#CUDA_VISIBLE_DEVICES=1 accelerate launch --gpu_ids 1 baselines/unimind/infer_topic_unimind.py \
#    --train_data_path data/DuRecDial/data/en_train.txt \
#    --dev_data_path data/DuRecDial/data/en_dev.txt \
#    --test_data_path data/DuRecDial/data/en_test.txt \
#    --tokenizer facebook/bart-base \
#    --plm_model facebook/bart-base \
#    --per_device_eval_batch_size 32 \
#    --max_sequence_length 512 \
#    --max_target_length 80 \
#    --goal_outpath ./unimind/goal/ \
#    --output_dir ./unimind/topic/ \
#    --seed 11
#
# #infer response
# CUDA_VISIBLE_DEVICES=1 accelerate launch --gpu_ids 1 baselines/unimind/infer_response_unimind.py \
#     --train_data_path data/DuRecDial/data/en_train.txt \
#     --dev_data_path data/DuRecDial/data/en_dev.txt \
#     --test_data_path data/DuRecDial/data/en_test.txt \
#     --tokenizer facebook/bart-base \
#     --plm_model facebook/bart-base \
#     --per_device_eval_batch_size 32 \
#     --max_sequence_length 512 \
#     --max_target_length 80 \
#     --goal_outpath ./unimind/goal/ \
#     --topic_outpath ./unimind/topic/ \
#     --output_dir ./unimind/response/ \
#     --seed 12
#


# infer goal
CUDA_VISIBLE_DEVICES=1 accelerate launch --gpu_ids 1 baselines/unimind/infer_goal_unimind.py \
    --dataset inspired \
    --train_data_path data/INSPIRED/data/dialog_data/train.tsv \
    --dev_data_path data/INSPIRED/data/dialog_data/dev.tsv \
    --test_data_path data/INSPIRED/data/dialog_data/test.tsv \
    --tokenizer facebook/bart-base \
    --plm_model facebook/bart-base \
    --per_device_eval_batch_size 32 \
    --max_sequence_length 512 \
    --max_target_length 80 \
    --output_dir ./unimind_inspired/goal/ \
    --seed 12

# #infer topic
CUDA_VISIBLE_DEVICES=1 accelerate launch --gpu_ids 1 baselines/unimind/infer_topic_unimind.py \
    --dataset inspired \
    --train_data_path data/INSPIRED/data/dialog_data/train.tsv \
    --dev_data_path data/INSPIRED/data/dialog_data/dev.tsv \
    --test_data_path data/INSPIRED/data/dialog_data/test.tsv \
    --tokenizer facebook/bart-base \
    --plm_model facebook/bart-base \
    --per_device_eval_batch_size 32 \
    --max_sequence_length 512 \
    --max_target_length 80 \
    --goal_outpath ./unimind_inspired/goal/ \
    --output_dir ./unimind_inspired/topic/ \
    --seed 11

 #infer response
 CUDA_VISIBLE_DEVICES=1 accelerate launch --gpu_ids 1 baselines/unimind/infer_response_unimind.py \
    --dataset inspired \
    --train_data_path data/INSPIRED/data/dialog_data/train.tsv \
    --dev_data_path data/INSPIRED/data/dialog_data/dev.tsv \
    --test_data_path data/INSPIRED/data/dialog_data/test.tsv \
     --tokenizer facebook/bart-base \
     --plm_model facebook/bart-base \
     --per_device_eval_batch_size 32 \
     --max_sequence_length 512 \
     --max_target_length 80 \
     --goal_outpath ./unimind_inspired/goal/ \
     --topic_outpath ./unimind_inspired/topic/ \
     --output_dir ./unimind_inspired/response/ \
     --seed 12

