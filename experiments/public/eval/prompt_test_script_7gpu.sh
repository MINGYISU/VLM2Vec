#!/bin/bash

# $1: query instruction prompt
if [ -z "$1" ]; then
    echo "No prompt provided, using default prompt_id: short_1"
    prompt_id="short_1"
else
    prompt_id="$1"
    echo "Using provied prompt: $prompt"
fi
export CUDA_VISIBLE_DEVICES="0,1,2,3"

echo "Available GPUs from CUDA perspective:"
python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}'); [print(f'Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

DATA_BASEDIR="/home/mingyi/sampled_video_frames"
OUTPUT_BASEDIR="/home/mingyi/AI-Projects/VLM2VEC_fork/VLM2Vec/prompt_test_output"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node=4 --master_port=2277 --max_restarts=0 eval.py \
    --pooling eos \
    --normalize true \
    --per_device_eval_batch_size 8 \
    --model_backbone qwen2_vl \
    --model_name VLM2Vec/VLM2Vec-V2.0 \
    --dataset_config experiments/public/eval/prompt_test.yaml \
    --query_instruction_prompt_file experiments/public/eval/prompt_list.yaml \
    --query_instruction_prompt_id "$prompt_id" \
    --encode_output_path "$OUTPUT_BASEDIR/prompt_test_gme_7b/$prompt_id" \
    --data_basedir "$DATA_BASEDIR"
