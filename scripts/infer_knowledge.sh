export CUDA_VISIBLE_DEVICES=2
#
#CUDA_VISIBLE_DEVICES=2 accelerate launch --gpu_ids 2 infer_knowledge.py \
#    --train_data_path data/DuRecDial/data/en_train.txt \
#    --dev_data_path data/DuRecDial/data/en_dev.txt \
#    --test_data_path data/DuRecDial/data/en_test.txt \
#    --tokenizer facebook/bart-base \
#    --plm_model facebook/bart-base \
#    --per_device_eval_batch_size 32 \
#    --max_sequence_length 512 \
#    --max_target_length 80 \
#    --goal_outpath ./policy_model/ \
#    --output_dir ./know_generation_model/ \
#    --seed 12


CUDA_VISIBLE_DEVICES=2 accelerate launch --gpu_ids 2 infer_knowledge.py \
    --dataset inspired \
    --train_data_path data/INSPIRED/data/dialog_data/train.tsv \
    --dev_data_path data/INSPIRED/data/dialog_data/dev.tsv \
    --test_data_path data/INSPIRED/data/dialog_data/test.tsv \
    --tokenizer facebook/bart-base \
    --plm_model facebook/bart-base \
    --per_device_eval_batch_size 32 \
    --max_sequence_length 512 \
    --max_target_length 80 \
    --goal_outpath ./policy_model_bert_4_inspired/ \
    --output_dir ./know_generation_model_inspired/ \
    --seed 12