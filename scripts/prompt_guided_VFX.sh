python prompt_guided_VFX.py \
    --prompt "Character is surrounded by puppies, Turn it into 3D cartoon style." \
    --image_or_video_path "/path/to/ref_image" \
    --model_path checkpoints/CogVideoX1.5-5B-I2V-OmniVFX \
    --output_path VFX.mp4 \
    --num_frames 81 \
    --width 1360 \
    --height 768 \
    --fps 16 \
    --generate_type "i2v"
