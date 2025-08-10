set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
workspace_dir=$(grep "WORKSPACE_DIR" ../config.py | cut -d "'" -f 2)
project_dir=$(grep "PROJECT_DIR" ../config.py | cut -d "'" -f 2)
bash ${workspace_dir}/env_run/train_bert/clear_gpu.sh
bash ${workspace_dir}/env_run/train_bert/stop_gpu.sh

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${project_dir}/verl/data/train.parquet \
    data.val_files=${project_dir}/verl/data/val.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=20480 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path={SFT_MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=23552 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='reasonrank' \
    trainer.experiment_name='grpo' \
    trainer.default_local_dir={YOUR_SAVING_PATH} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=100 \
    trainer.total_epochs=1 $@


current_dir=$PWD
cd ${workspace_dir}/env_run/train_bert/
bash train_bert.sh
cd $current_dir