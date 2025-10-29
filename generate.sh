# DES on Stable Diffusion v1.5
python generate.py \
    --model_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --device cuda:1 \
    --prompts_csv "datasets/coco_prompts.csv" \
    --output_path "t2i_coco" \
    --start_idx 0 \
    --end_idx 128 \
    --training_method des \
    --text_encoder_path "checkpoints/des_copro_sexual/1e-05_0.3/checkpoint-2.pt"
