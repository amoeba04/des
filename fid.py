import argparse
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import glob
import re
import numpy as np
import gc
from transformers import BlipProcessor, BlipForConditionalGeneration
import json

def extract_prompt_number(filename):
    """Extract prompt number (Np) from generated image filename"""
    pattern = r'_(\d+)p_'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return None

def extract_prompt_number_tmp(filename):
    """Extract prompt number (Np) from generated image filename"""
    pattern = r'_(\d+)p'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return None

def find_matching_gen_image(original_csv_index, gen_imgs_dir):
    """
    Find generated image matching the given index in the original CSV file

    Args:
        original_csv_index: 0-based index in the original COCO CSV file
        gen_imgs_dir: Directory containing generated images

    Returns:
        Path to the matching generated image or None if not found
    """
    all_files = glob.glob(os.path.join(gen_imgs_dir, "*.png"))

    # CSV file row index starts from 0, filename also starts from 0p
    # (0 -> 0p, 1 -> 1p, etc.)
    prompt_num = original_csv_index

    for file_path in all_files:
        filename = os.path.basename(file_path)
        extracted_num = extract_prompt_number(filename)
        if extracted_num is None:
            extracted_num = extract_prompt_number_tmp(filename)
        if extracted_num == prompt_num:
            return file_path

    return None

def find_matching_original_image(case_number, coco_imgs_dir):
    """Find original image matching the given case number"""
    all_files = glob.glob(os.path.join(coco_imgs_dir, "*.jpg"))
    case_number_str = str(case_number).zfill(12)  # Ensure proper padding

    for file_path in all_files:
        if case_number_str in file_path:
            return file_path

    return None

def calculate_clip_score(gen_imgs_dir, prompts, case_numbers, original_indices, clip_model, clip_processor, device):
    """Calculate CLIP score between generated images and prompts using proper matching"""
    model = clip_model
    processor = clip_processor

    clip_scores = []
    valid_pairs = 0

    for i, (prompt, case_number, original_idx) in enumerate(zip(prompts, case_numbers, original_indices)):
        # Find matching generated image using the original CSV index
        gen_img_path = find_matching_gen_image(original_idx, gen_imgs_dir)

        prompt_num = original_idx  # CSV row index = prompt number

        if gen_img_path and os.path.exists(gen_img_path):
            image = Image.open(gen_img_path)

            inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                score = logits_per_image.item()
                clip_scores.append(score)
                valid_pairs += 1
                if i % 10000 == 0:
                    print(f"CLIP Score for prompt {prompt_num}p ({i+1}/{len(prompts)}): {score:.4f}")
        else:
            print(f"Warning: Matching generated image not found for prompt {prompt_num}p (original idx {original_idx})")

    if clip_scores:
        average_clip_score = sum(clip_scores) / len(clip_scores)
        return average_clip_score, clip_scores, valid_pairs
    else:
        return 0, [], 0

def calculate_pickscore(gen_imgs_dir, prompts, case_numbers, original_indices, pick_model, pick_processor, device):
    """Calculate PickScore between generated images and prompts using proper matching"""
    model = pick_model
    processor = pick_processor

    pick_scores = []
    valid_pairs = 0

    for i, (prompt, case_number, original_idx) in enumerate(zip(prompts, case_numbers, original_indices)):
        # Find matching generated image using the original CSV index
        gen_img_path = find_matching_gen_image(original_idx, gen_imgs_dir)

        prompt_num = original_idx  # CSV row index = prompt number

        if gen_img_path and os.path.exists(gen_img_path):
            image = Image.open(gen_img_path)

            # Preprocess image and text
            image_inputs = processor(
                images=[image],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)

            text_inputs = processor(
                text=[prompt],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)

            # Calculate PickScore
            with torch.no_grad():
                # Get embeddings
                image_embs = model.get_image_features(**image_inputs)
                image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

                text_embs = model.get_text_features(**text_inputs)
                text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

                # Calculate score
                score = model.logit_scale.exp() * (text_embs @ image_embs.T)[0][0]
                pick_scores.append(score.item())
                valid_pairs += 1

                if i % 10000 == 0:
                    print(f"PickScore for prompt {prompt_num}p ({i+1}/{len(prompts)}): {score.item():.4f}")
        else:
            print(f"Warning: Matching generated image not found for prompt {prompt_num}p (original idx {original_idx})")

    if pick_scores:
        average_pick_score = sum(pick_scores) / len(pick_scores)
        return average_pick_score, pick_scores, valid_pairs
    else:
        return 0, [], 0



def calculate_blipscore(gen_imgs_dir, prompts, case_numbers, original_indices, clip_model, clip_processor, blip_model, blip_processor, device):
    """Calculate BLIP score between generated images and prompts using transformers models"""
    blip_scores = []
    blip_captions = []
    valid_pairs = 0

    # Batch processing for memory management
    batch_size = 1

    for i, (prompt, case_number, original_idx) in enumerate(zip(prompts, case_numbers, original_indices)):
        # Periodic memory cleanup
        if i % 10 == 0:
            torch.cuda.empty_cache()

        # Find matching generated image using the original CSV index
        gen_img_path = find_matching_gen_image(original_idx, gen_imgs_dir)

        prompt_num = original_idx  # CSV row index = prompt number

        if gen_img_path and os.path.exists(gen_img_path):
            try:
                # 1. Load image
                image = Image.open(gen_img_path).convert("RGB")

                # 2. Generate caption with BLIP
                with torch.no_grad():
                    inputs = blip_processor(images=image, return_tensors="pt").to(device)
                    generated_ids = blip_model.generate(**inputs, max_length=50)
                    blip_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    blip_captions.append(blip_caption)

                # 3. Extract prompt and caption embeddings with CLIP
                with torch.no_grad():
                    # Prompt embedding
                    prompt_inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(device)
                    prompt_outputs = clip_model.get_text_features(**prompt_inputs)
                    prompt_embedding = prompt_outputs / prompt_outputs.norm(dim=1, keepdim=True)

                    # Caption embedding
                    caption_inputs = clip_processor(text=[blip_caption], return_tensors="pt", padding=True).to(device)
                    caption_outputs = clip_model.get_text_features(**caption_inputs)
                    caption_embedding = caption_outputs / caption_outputs.norm(dim=1, keepdim=True)

                # 4. Calculate cosine similarity
                similarity = torch.nn.functional.cosine_similarity(prompt_embedding, caption_embedding).item()
                blip_scores.append(similarity)
                valid_pairs += 1

                if i % 10000 == 0:
                    print(f"BLIP Score for prompt {prompt_num}p ({i+1}/{len(prompts)}): {similarity:.4f}")
                    print(f"  Prompt: {prompt[:50]}...")
                    print(f"  Caption: {blip_caption}")

                # Memory cleanup
                del prompt_inputs, prompt_outputs, prompt_embedding
                del caption_inputs, caption_outputs, caption_embedding
                del inputs, generated_ids
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error calculating BLIP Score for prompt {prompt_num}p: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Warning: Matching generated image not found for prompt {prompt_num}p (original idx {original_idx})")

    if blip_scores:
        average_blip_score = sum(blip_scores) / len(blip_scores)
        return average_blip_score, blip_scores, blip_captions, valid_pairs
    else:
        return 0, [], [], 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'FID_Eval',
                    description = 'Evaluate FID score')

    parser.add_argument('--job', help='calculate evaluation metrics', type=str, required=False, default='fid',
                       choices=['fid', 'pickscore', 'blipscore'])
    parser.add_argument('--gen_imgs_path', help='generated image folder for evaluation', type=str, required=True)
    parser.add_argument('--coco_imgs_path', help='coco real image folder for evaluation', type=str, required=False, default='datasets/coco_10k')
    parser.add_argument('--prompt_path', help='prompt CSV file', type=str, required=False, default='datasets/coco_10k.csv')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run the model on (e.g., "cuda:0" or "cpu")')
    parser.add_argument('--output_file', help='output file name', type=str, required=False, default=None)

    args = parser.parse_args()
    device = args.device

    # Set filename to save results
    if args.output_file is None:
        args.output_file = args.gen_imgs_path + f'_{args.job}.txt'

    if args.job == 'fid':
        from T2IBenchmark import calculate_fid
        fid, _ = calculate_fid(args.gen_imgs_path, args.coco_imgs_path, device=device)

        content = f'FID={fid}'
        file_path = args.output_file
        print(fid)

    elif args.job == 'pickscore':
        # Import necessary libraries for PickScore
        from transformers import AutoProcessor, AutoModel

        # Load PickScore model and processor
        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

        processor = AutoProcessor.from_pretrained(processor_name_or_path)
        model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

        # load CSV
        csv_file_path = args.prompt_path
        df = pd.read_csv(csv_file_path)

        # Check and extract required data
        prompts = df['prompt'].tolist() if 'prompt' in df.columns else df[0].tolist()
        case_numbers = df['case_number'].tolist() if 'case_number' in df.columns else None

        if case_numbers is None:
            print("Warning: 'case_number' column not found in CSV. Using indices as case numbers.")
            case_numbers = [i for i in range(len(prompts))]

        # Generate original indices
        original_indices = list(range(len(prompts)))

        # Calculate improved PickScore
        print(f"Calculating PickScores for {len(prompts)} prompts...")
        average_pick_score, pick_scores, valid_pairs = calculate_pickscore(
            args.gen_imgs_path,
            prompts,
            case_numbers,
            original_indices,
            model,
            processor,
            device
        )

        # Save results
        content = f'Mean PickScore = {average_pick_score}\nValid image-prompt pairs: {valid_pairs}/{len(prompts)}'
        file_path = args.output_file

        # Add PickScore distribution
        if pick_scores:
            pick_scores = np.array(pick_scores)
            content += f'\nPickScore Min/Median/Max: {np.min(pick_scores):.4f}/{np.median(pick_scores):.4f}/{np.max(pick_scores):.4f}'
            content += f'\nPickScore Std: {np.std(pick_scores):.4f}'

    elif args.job == 'blipscore':
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()

        # Load CLIP model
        print("Loading CLIP model...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()

        # Load BLIP model
        print("Loading BLIP model...")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model.eval()

        # Load CSV file
        csv_file_path = args.prompt_path
        df = pd.read_csv(csv_file_path)

        # Check and extract required data
        prompts = df['prompt'].tolist() if 'prompt' in df.columns else df[0].tolist()
        case_numbers = df['case_number'].tolist() if 'case_number' in df.columns else None

        if case_numbers is None:
            print("Warning: 'case_number' column not found in CSV. Using indices as case numbers.")
            case_numbers = [i for i in range(len(prompts))]

        # Generate original indices
        original_indices = list(range(len(prompts)))

        # Calculate BLIP Score - calculate for all prompts
        print(f"Calculating BLIP scores for all {len(prompts)} prompts...")

        # Additional memory cleanup before calculation
        torch.cuda.empty_cache()
        gc.collect()

        average_blip_score, blip_scores, blip_captions, valid_pairs = calculate_blipscore(
            args.gen_imgs_path,
            prompts,
            case_numbers,
            original_indices,
            clip_model,
            clip_processor,
            blip_model,
            blip_processor,
            device
        )

        # Save results - concise output without sample captions
        content = f'Mean BLIP Score = {average_blip_score}\nValid image-prompt pairs: {valid_pairs}/{len(prompts)}'
        file_path = args.output_file

        # Add BLIP Score distribution
        if blip_scores:
            blip_scores = np.array(blip_scores)
            content += f'\nBLIP Score Min/Median/Max: {np.min(blip_scores):.4f}/{np.median(blip_scores):.4f}/{np.max(blip_scores):.4f}'
            content += f'\nBLIP Score Std: {np.std(blip_scores):.4f}'

            # Save caption and score data to JSON file (sample caption output removed)
            captions_data = {
                'average_score': average_blip_score,
                'valid_pairs': valid_pairs,
                'std': float(np.std(blip_scores)),
                'min': float(np.min(blip_scores)),
                'median': float(np.median(blip_scores)),
                'max': float(np.max(blip_scores))
            }
            json_file_path = args.output_file.replace('.txt', '_stats.json')
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(captions_data, json_file, indent=2)
            print(f"BLIP score statistics saved to: {json_file_path}")

        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()

    print(content)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
