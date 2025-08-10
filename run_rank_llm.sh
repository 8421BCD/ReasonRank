# export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
workspace_dir=$(grep "WORKSPACE_DIR" config.py | cut -d "'" -f 2)

# all datasets:
# DATASETS=('dl19' 'dl20' 'covid' 'dbpedia' 'scifact' 'nfcorpus' 'signal' 'robust04' 'news')
# DATASETS=('economics' 'earth_science' 'robotics' 'biology' 'psychology' 'stackoverflow' 'sustainable_living' 'leetcode' 'pony' 'aops' 'theoremqa_questions' 'theoremqa_theorems')
# DATASETS=('r2med_Biology' 'r2med_Bioinformatics' 'r2med_Medical-Sciences' 'r2med_MedXpertQA-Exam' 'r2med_MedQA-Diag' 'r2med_PMC-Treatment' 'r2med_PMC-Clinical' 'r2med_IIYi-Clinical')
################ evaluate reasonrank-7B using ReasonIR retrieval #################
window_size=20
model_name=reasonrank-7B
DATASETS=('economics')
python run_rank_llm.py \
    --model_path ${workspace_dir}/trained_models/${model_name} \
    --window_size $window_size \
    --step_size 10 \
    --retrieval_num 100 \
    --num_passes 1 \
    --reasoning_maxlen 3072 \
    --retrieval_method reasonir \
    --use_gpt4cot_retrieval True \
    --datasets ${DATASETS[@]} \
    --shuffle_candidates False \
    --prompt_mode rank_GPT_reasoning \
    --context_size 32768 \
    --variable_passages \
    --vllm_batched True \
    --batch_size 512 \
    --output "${model_name}.txt" \
    --num_gpus 4 \
    --notes ""

################ evaluate reasonrank-7B using custom retrieval #################
window_size=20
model_name=reasonrank-7B
DATASETS=('economics')
python run_rank_llm.py \
    --model_path ${workspace_dir}/trained_models/${model_name} \
    --retrieval_results_name custom.txt \
    --window_size $window_size \
    --step_size 10 \
    --retrieval_num 100 \
    --num_passes 1 \
    --reasoning_maxlen 3072 \
    --datasets ${DATASETS[@]} \
    --shuffle_candidates False \
    --prompt_mode rank_GPT_reasoning \
    --context_size 32768 \
    --vllm_batched True \
    --batch_size 512 \
    --output "${model_name}.txt" \
    --num_gpus 4 \
    --notes ""

################ evaluate reasonrank-32B using ReasonIR retrieval #################
window_size=20
model_name=reasonrank-32B
DATASETS=('economics')
python run_rank_llm.py \
    --model_path ${workspace_dir}/trained_models/${model_name} \
    --lora_path ${workspace_dir}/trained_models/${model_name}/lora_adapter \
    --window_size $window_size \
    --step_size 10 \
    --retrieval_num 100 \
    --num_passes 1 \
    --reasoning_maxlen 3072 \
    --retrieval_method reasonir \
    --use_gpt4cot_retrieval True \
    --datasets ${DATASETS[@]} \
    --shuffle_candidates False \
    --prompt_mode rank_GPT_reasoning \
    --context_size 32768 \
    --vllm_batched True \
    --batch_size 512 \
    --output "${model_name}.txt" \
    --num_gpus 4 \
    --notes ""
