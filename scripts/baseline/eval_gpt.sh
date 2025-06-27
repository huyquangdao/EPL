export CUDA_VISIBLE_DEVICES=4

CUDA_VISIBLE_DEVICES=4 accelerate launch --gpu_ids 4 baselines/dialoggpt/infer_response_dialoggpt.py \
    --train_data_path data/DuRecDial/data/en_train.txt \
    --dev_data_path data/DuRecDial/data/en_dev.txt \
    --test_data_path data/DuRecDial/data/en_test.txt \
    --tokenizer gpt2 \
    --plm_model gpt2 \
    --per_device_eval_batch_size 32 \
    --max_sequence_length 512 \
    --max_target_length 80 \
    --output_dir ./gpt2_model/ \
    --seed 12