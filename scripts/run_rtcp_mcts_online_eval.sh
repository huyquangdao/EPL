export CUDA_VISIBLE_DEVICES=4
seed=1

CUDA_VISIBLE_DEVICES=4 accelerate launch --gpu_ids 4 online_evaluation.py \
    --train_data_path data/DuRecDial/data/en_train.txt \
    --dev_data_path data/DuRecDial/data/en_dev.txt \
    --test_data_path data/DuRecDial/data/en_test.txt \
    --policy_tokenizer bert-base-cased \
    --plm_policy_model bert-base-cased \
    --generation_tokenizer facebook/bart-base \
    --plm_generation_model facebook/bart-base \
    --know_generation_tokenizer facebook/bart-base \
    --plm_know_generation_model facebook/bart-base \
    --rollouts 10 \
    --num_items 100 \
    --alg p_uct \
    --hidden_size 128 \
    --lm_size 768 \
    --target_set_path ./target_set_${seed}/ \
    --memory_path self_simulations_${seed}.txt \
    --horizon 5 \
    --max_sequence_length 512 \
    --max_gen_length 50 \
    --policy_model_path ./rtcp/ \
    --generation_model_path ./generation_model/ \
    --know_generation_model_path ./know_generation_model/ \
    --seed ${seed} \
    --use_rtcp_policy \
    --lm_size 768 \
    --ffn_size 3072 \
    --n_layers 12 \
    --n_heads 8 \
    --fc_size 128