backend="fsdp"
checkpoint_dir=$1
hf_model_path=$2
target_dir=$3


python {YOUR_PROJECT_DIR}/verl/scripts/legacy_model_merger.py merge \
    --backend $backend \
    --local_dir $checkpoint_dir \
    --target_dir $target_dir
