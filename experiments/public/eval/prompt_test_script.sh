#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

echo "Available GPUs after CUDA_VISIBLE_DEVICES setting:"
nvidia-smi --query-gpu=index,name --format=csv,noheader

DATA_BASEDIR="/home/mingyi/sampled_video_frames"
OUTPUT_BASEDIR="/home/mingyi/AI-Projects/VLM2VEC_fork/VLM2Vec/prompt_test_output"

cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES 
    python eval.py \
    --pooling eos \
    --normalize true \
    --per_device_eval_batch_size 16 \
    --model_backbone qwen2_vl \
    --model_name VLM2Vec/VLM2Vec-V2.0 \
    --dataset_config experiments/public/eval/prompt_test.yaml \
    --encode_output_path $OUTPUT_BASEDIR/result10 \
    --data_basedir $DATA_BASEDIR"

eval "$cmd"