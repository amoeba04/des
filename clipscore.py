import torch
from PIL import Image
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
import os
from tqdm import tqdm
import argparse
import re

# Prevent tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def compute_clip_score(image, prompt, model, processor, device):
    """
    Compute CLIP Score for a single image and prompt.

    Args:
        image: PIL Image object
        prompt: Text prompt string
        model: CLIP model
        processor: CLIP processor
        device: Device to use for computation

    Returns:
        float: CLIP Score (cosine similarity * 100)
    """
    # Prepare CLIP inputs
    inputs = processor(
        text=[prompt],
        images=image,
        return_tensors="pt",
        padding=True
    )

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Extract image and text features
    with torch.no_grad():
        # Extract and normalize image features
        img_features = model.get_image_features(inputs['pixel_values'])
        img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

        # Extract and normalize text features
        txt_features = model.get_text_features(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)

        # Compute CLIP score: 100 * cosine_similarity
        clip_score = 100 * (img_features * txt_features).sum(dim=-1)

    return clip_score.cpu().item()

def calculate_clip_scores(image_folder: str, csv_file: str, device: str) -> pd.DataFrame:
    """
    Calculate CLIP Scores for all images in a folder.

    Args:
        image_folder: Path to the folder containing images
        csv_file: Path to CSV file with text prompts
        device: Device to use for computation (e.g., 'cuda:0', 'cpu')
    Returns:
        pandas.DataFrame: DataFrame containing computed CLIP Scores
    """
    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Ensure model is in eval mode
    model.eval()

    # List to store results
    clip_scores = []

    # Read CSV file
    if csv_file:
        df = pd.read_csv(csv_file)
        # Use 'prompt' column if exists, otherwise use first column
        # Header is automatically processed (first row is treated as header)
        prompts = df['prompt'].tolist() if 'prompt' in df.columns else df.iloc[:, 0].tolist()
    else:
        raise ValueError("CSV file is required to determine prompts")

    # Create and sort list of image files (filter PNG files only)
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    print(f"Found {len(image_files)} image files in directory.")

    # Process each image file
    for image_file in tqdm(image_files):
        # Extract prompt index from filename
        match = re.search(r'_(\d+)p\.', image_file)
        if not match:
            match = re.search(r'(\d+)p\.', image_file)

        if not match:
            print(f"Warning: Cannot extract prompt index from filename: {image_file}. Skipping file.")
            continue

        # Prompt index extracted from filename (e.g., 0p -> 0, 1p -> 1)
        prompt_idx = int(match.group(1))

        # CSV index mapping: filename Np corresponds to CSV index N
        # (0p.png -> prompts[0], 1p.png -> prompts[1], ...)
        csv_idx = prompt_idx

        # Check if index is valid
        if csv_idx >= len(prompts):
            print(f"Warning: Prompt index {csv_idx} out of range (max: {len(prompts)-1}). Skipping {image_file}.")
            continue

        image_path = os.path.join(image_folder, image_file)

        # Load image
        image = Image.open(image_path)

        # Get prompt from CSV at the corresponding index
        prompt = prompts[csv_idx]

        # Compute CLIP Score
        clip_score = compute_clip_score(image, prompt, model, processor, device)

        clip_scores.append({
            'file_name': image_file,
            'prompt': prompt,
            'clip_score': clip_score,
            'prompt_idx': prompt_idx
        })

    if not clip_scores:
        raise ValueError("No valid images were processed. Check your image folder and CSV file.")

    # Convert results to DataFrame
    results_df = pd.DataFrame(clip_scores)

    # Sort by file_name
    results_df = results_df.sort_values(by='file_name')

    # Calculate average CLIP score
    avg_clip_score = results_df['clip_score'].mean()

    print(f"Average CLIP Score: {avg_clip_score:.4f}")
    print(f"Processed {len(clip_scores)} images out of {len(prompts)} prompts.")

    return results_df

def main():
    parser = argparse.ArgumentParser(description='Calculate CLIP scores for generated images')
    parser.add_argument('--image_folder', type=str, required=True,
                        help='Path to the folder containing generated images')
    parser.add_argument('--csv_file', type=str, default=None,
                        help='csv file to read text prompts')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run the model on (e.g., "cuda" or "cpu")')

    args = parser.parse_args()

    calculate_clip_scores(args.image_folder, args.csv_file, args.device)

if __name__ == "__main__":
    main()
