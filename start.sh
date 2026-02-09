#!/bin/bash

# consider use double loop to make it elegant
for cur_prompt_id in "distractor_3" "distractor_2" "distractor_1" "rephrased_1" "rephrased_2" "rephrased_3" "short_1" "short_2" "short_3" "long_1" "long_2" "long_3"
do
    # bash experiments/public/eval/prompt_test_script_7gpu.sh "$cur_prompt_id"
    bash experiments/public/eval/run_visdoc_prompt_test_1gpu.sh "$cur_prompt_id"
done