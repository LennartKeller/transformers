export CUDA_VISIBLE_DEVICES="1"

python run_summarization.py \
    --model_name_or_path "pyramidion2pyramidion-3layer" \
    --tokenizer_name "pyramidion2pyramidion-3layer" \
    --lang "de" \
    --dataset_name "mlsum" \
    --dataset_config_name "de" \
    --text_column "text" \
    --summary_column "summary" \
    --resize_position_embeddings false \
    --max_source_length 512 \
    --max_target_length 512 \
    --val_max_target_length 256 \
    --pad_to_max_length true \
    --output_dir "pyramidion-3layer-mlsum" \
    --overwrite_output_dir true \
    --preprocessing_num_workers 12 \
    --do_train true \
    --do_eval true \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --logging_dir "pyramidion-3layer-mlsum/logts" \
    --logging_strategy "steps" \
    --logging_first_step true \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_total_limit 10 \
    --save_steps 5000 \
    --seed 42 \
    --predict_with_generate true \



