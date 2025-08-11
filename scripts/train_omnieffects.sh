export MODEL_NAME="THUDM/CogVideoX-5b-I2V"
export DATASET_META_NAME="Omni-VFX/VFX_data.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

PORT=$MASTER_PORT NODE_RANK=$RANK NNODES=$WORLD_SIZE \
accelerate launch --config_file config/accelerate/accelerate_deepspeed_8gpu.yaml scripts/train_omnieffects.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --mixed_precision="bf16" \
  --vae_mini_batch=1 \
  --enable_slicing \
  --enable_tiling \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_height=480 \
  --video_sample_width=720 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --dataloader_num_workers=2 \
  --allow_tf32 \
  --learning_rate=1e-04 \
  --adam_weight_decay=1e-4 \
  --adam_epsilon=1e-8 \
  --adam_beta1=0.9 \
  --adam_beta2=0.95 \
  --max_grad_norm=1.0 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=20 \
  --num_train_epochs=100 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --noised_image_drop_ratio=0.05 \
  --checkpointing_steps=2 \
  --output_dir="output" \
  --seed=42 \
  --rank=128 \
  --lora_weight=1 \
  --network_alpha=128 \
  --random_crop_prob=0.0 \
  --num_experts=4 \
  --top_k=1 \
  --cond_linear \
  --mha_lora \
  --aux_loss \
  --m2v_mask \
  --moe_lora \
  --cond_lora \
  --num_cond=1 \
  