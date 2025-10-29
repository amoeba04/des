# FID
python fid.py \
    --gen_imgs_path results/des/stable-diffusion-v1-5/t2i_coco/ \
    --coco_imgs_path datasets/coco_10k/ \
    --device cuda:0

# CLIPScore
python clipscore.py \
    --image_folder results/des/stable-diffusion-v1-5/t2i_coco/ \
    --csv_file datasets/coco_prompts.csv \
    --device cuda:0

# ASR (NudeNet)
python tasks/img_batch_classify.py --job nudity --cls_class nudity --folder_dir results/des/stable-diffusion-v1-5/t2i_sneaky/

python tasks/img_batch_classify.py --job nudity --cls_class nudity --folder_dir results/des/stable-diffusion-v1-5/t2i_mma/

python tasks/img_batch_classify.py --job nudity --cls_class nudity --folder_dir results/des/stable-diffusion-v1-5/t2i_ringabell/

python tasks/img_batch_classify.py --job nudity --cls_class nudity --folder_dir results/des/stable-diffusion-v1-5/t2i_p4d/

python tasks/img_batch_classify.py --job nudity --cls_class nudity --folder_dir results/des/stable-diffusion-v1-5/t2i_i2p/
