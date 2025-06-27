export CUDA_VISIBLE_DEVICES=5
seed=4

## for duredial
#CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 infer_tcp_policy.py \
#    --dataset inspired \
#    --train_data_path data/INSPIRED/data/dialog_data/train.tsv \
#    --dev_data_path data/INSPIRED/data/dialog_data/dev.tsv \
#    --test_data_path data/INSPIRED/data/dialog_data/test.tsv \
#    --tokenizer bert-base-cased \
#    --max_sequence_length 512 \
#    --plm_model bert-base-cased \
#    --hidden_size 128 \
#    --lm_size 768 \
#    --output_dir ./policy_model_bert_${seed}_inspired/ \
#    --seed ${seed}
#
#
CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 infer_policy.py \
    --dataset inspired \
    --train_data_path data/INSPIRED/data/dialog_data/train.tsv \
    --dev_data_path data/INSPIRED/data/dialog_data/dev.tsv \
    --test_data_path data/INSPIRED/data/dialog_data/test.tsv \
    --tokenizer bert-base-cased \
    --max_sequence_length 512 \
    --plm_model bert-base-cased \
    --hidden_size 128 \
    --lm_size 768 \
    --output_dir ./policy_model_bert_${seed}_inspired/ \
    --seed ${seed}