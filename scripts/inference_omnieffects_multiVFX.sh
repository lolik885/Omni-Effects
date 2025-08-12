python scripts/inference_omnieffects.py \
    --model_path "THUDM/CogVideoX-5b-I2V" \
    --lora_path "checkpoints/multi-VFX" \
    --lora_name "lora_adapter" \
    --output "output" \
    --data_path dataset/test/test3.json \
    --device_id 0 \
    --cond_linear \
    --num_cond 2 \
    --mha_lora \
    --num_experts 4 \
    --top_k 4 \
    --m2v_mask

python scripts/inference_omnieffects.py \
    --model_path "THUDM/CogVideoX-5b-I2V" \
    --lora_path "checkpoints/multi-VFX" \
    --lora_name "lora_adapter" \
    --output "output" \
    --data_path dataset/test/test3.json \
    --device_id 0 \
    --cond_linear \
    --num_cond 3 \
    --mha_lora \
    --num_experts 4 \
    --top_k 4 \
    --m2v_mask
