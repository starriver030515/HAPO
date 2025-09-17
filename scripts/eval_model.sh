# Merge training checkpoint
python3 model_merger.py merge \
    --backend fsdp \
    --local_dir $YOUR_CKPT_PATH/actor \
    --target_dir $YOUR_MODEL_PATH/model

# Generate responses
python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=valid.parquet \
    data.prompt_key=prompt \
    data.batch_size=1024 \
    data.n_samples=1 \
    data.output_path=$OUTPUT_DIR/results.parquet \
    model.path=$YOUR_MODEL_PATH/model \
    rollout.temperature=0.5 \
    rollout.top_p=0.7 \
    rollout.prompt_length=1024 \
    rollout.response_length=10240 \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.9 \
    rollout.max_num_batched_tokens=65536

# Evaluation
python3 -m verl.trainer.main_eval \
    data.path=$OUTPUT_DIR/results.parquet \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=../verl/utils/reward_score/__init__.py \
    custom_reward_function.name=default_compute_score
