export CUDA_VISIBLE_DEVICES=5
seed=1

# CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 self_simulation.py \
#     --dataset durecdial \
#    --train_data_path data/DuRecDial/data/en_train.txt \
#    --dev_data_path data/DuRecDial/data/en_dev.txt \
#    --test_data_path data/DuRecDial/data/en_test.txt \
#    --policy_tokenizer bert-base-cased \
#    --plm_policy_model bert-base-cased \
#    --generation_tokenizer facebook/bart-base \
#    --plm_generation_model facebook/bart-base \
#    --know_generation_tokenizer facebook/bart-base \
#    --plm_know_generation_model facebook/bart-base \
#    --rollouts 5 \
#    --top_k 5 \
#    --epsilon 0.1 \
#    --num_items 376 \
#    --alg p_uct \
#    --hidden_size 128 \
#    --lm_size 768 \
#    --target_set_path ./target_set_full_${seed}/ \
#    --memory_path self_simulation_full.txt \
#    --horizon 5 \
#    --max_sequence_length 512 \
#    --max_gen_length 50 \
#    --policy_model_path ./policy_model_bert_4/ \
#    --generation_model_path ./generation_model/ \
#    --know_generation_model_path ./know_generation_model/ \
#    --seed ${seed}


CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 self_simulation.py \
    --dataset inspired \
    --train_data_path data/INSPIRED/data/dialog_data/train.tsv \
    --dev_data_path data/INSPIRED/data/dialog_data/dev.tsv \
    --test_data_path data/INSPIRED/data/dialog_data/test.tsv \
    --policy_tokenizer bert-base-cased \
    --plm_policy_model bert-base-cased \
    --generation_tokenizer facebook/bart-base \
    --plm_generation_model facebook/bart-base \
    --know_generation_tokenizer facebook/bart-base \
    --plm_know_generation_model facebook/bart-base \
    --rollouts 10 \
    --top_k 5 \
    --epsilon 0.1 \
    --num_items 55 \
    --alg p_uct \
    --hidden_size 128 \
    --lm_size 768 \
    --target_set_path ./target_set_inspired_${seed}/ \
    --memory_path memory_results.txt \
    --horizon 5 \
    --max_sequence_length 512 \
    --max_gen_length 50 \
    --policy_model_path ./policy_model_bert_4_inspired/ \
    --generation_model_path ./generation_model_inspired/ \
    --know_generation_model_path ./know_generation_model_inspired/ \
    --seed ${seed}