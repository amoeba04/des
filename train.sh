# save codebook
python save_codebook.py \
    --model_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --device cuda:0 \
    --csv_path datasets/safe_prompts_copro_sexual.csv \
    --save_dir codebook_copro_sexual

# train des
python train_des.py \
    --model_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --device cuda:0 \
    --codebook_dir codebook_copro_sexual \
    --unsafe_csv_path datasets/unsafe_prompts_copro_sexual.csv \
    --safe_csv_path datasets/safe_prompts_copro_sexual.csv \
    --output_dir checkpoints/des_copro_sexual \
    --num_epochs 2 \
    --learning_rate 1e-5 \
    --batch_size 128 \
    --lambda_safe 0.3 \
    --save_every 1 \
    --ablation 1 2 3 \
    --concept_prompt "nudity" \
    --concept_guidance_scale 200.0 \
    --safe_embedding_path checkpoints/des_copro_sexual/safe_embeddings.pth

# save codebook (another text encoder for SD v3.5)
python save_codebook.py \
    --model_path stabilityai/stable-diffusion-3.5-medium \
    --device cuda:0 \
    --csv_path datasets/safe_prompts_copro_sexual.csv \
    --save_dir codebook_sdv3_copro_sexual \
    --encoder_idx 3

# train des (another text encoder for SD v3.5)
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_des.py \
    --model_path stabilityai/stable-diffusion-3.5-medium \
    --codebook_dir codebook_sdv3_copro_sexual \
    --unsafe_csv_path datasets/unsafe_prompts_copro_sexual.csv \
    --safe_csv_path datasets/safe_prompts_copro_sexual.csv \
    --output_dir checkpoints/des_sdv3_sexual_encoder3 \
    --num_epochs 2 \
    --learning_rate 1e-5 \
    --batch_size 4 \
    --lambda_safe 0.3 \
    --save_every 1 \
    --ablation 1 2 3 \
    --concept_prompt "nudity" \
    --concept_guidance_scale 200.0 \
    --safe_embedding_path checkpoints/des_sdv3_sexual_encoder3/safe_embeddings.pth \
    --text_encoder_idx 3 \
    --multi_gpu