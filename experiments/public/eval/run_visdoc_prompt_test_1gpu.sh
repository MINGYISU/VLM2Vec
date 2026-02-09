#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

echo "Available GPUs from CUDA perspective:"
python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}'); [print(f'Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

if [ -z "$1" ]; then
    echo "No prompt provided, using default prompt_id: short_1"
    prompt_id="short_1"
else
    prompt_id="$1"
    echo "Using provided prompt: $prompt"
fi

DATA_BASEDIR="/data/mingyi/MMEB/visdoc/images"
OUTPUT_BASEDIR="/home/mingyi/AI-Projects/VLM2VEC_fork/VLM2Vec/prompt_test_output"
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python eval.py \
    --pooling eos \
    --normalize true \
    --per_device_eval_batch_size 8 \
    --model_backbone qwen2_vl \
    --model_name VLM2Vec/VLM2Vec-V2.0 \
    --dataset_config experiments/public/eval/visdoc.yaml \
    --encode_output_path "$OUTPUT_BASEDIR/prompt_visdoc_run/$prompt_id" \
    --data_basedir "$DATA_BASEDIR" \
    --query_instruction_prompt_file experiments/public/eval/prompt_list_vd.yaml \
    --query_instruction_prompt_id "$prompt_id" \
