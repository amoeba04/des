import torch
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import AutoModelForCausalLM
from PIL import Image
import argparse
import pandas as pd
import os
from tqdm import tqdm
import faiss
import numpy as np
from dataclasses import dataclass

@dataclass
class CLIPEmbeddingInfo:
    prompt: str
    embedding: np.ndarray  # [seq_len, hidden_dim]

class CLIPEmbeddingCodebook:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.embeddings = []
        self.index = None
    
    def add_embedding(self, prompt: str, embedding: torch.Tensor):
        """Add CLIP text embedding to codebook
        Args:
            prompt: Text prompt
            embedding: CLIP text embedding of shape [seq_len, hidden_dim]
        """
        # Store embedding exactly as it comes from the model
        self.embeddings.append(CLIPEmbeddingInfo(
            prompt=prompt,
            embedding=embedding.cpu().numpy()  # Convert to numpy but preserve shape
        ))
    
    def build_index(self):
        """Build FAISS index for similarity search"""
        # Flatten embeddings for FAISS
        embeddings = np.stack([
            info.embedding.reshape(-1) for info in self.embeddings
        ])  # [num_prompts, seq_len * hidden_dim]

        dimension = embeddings.shape[1]

        # Create CPU index
        self.index = faiss.IndexFlatIP(dimension)
        print(f"Created CPU FAISS index with dimension {dimension}")

        # Normalize embeddings
        embeddings_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Prevent division by zero
        embeddings_norm = np.maximum(embeddings_norm, 1e-8)
        embeddings = embeddings / embeddings_norm

        self.index.add(embeddings.astype(np.float32))
    
    def save(self, save_dir: str, max_len: int = None):
        """Save codebook and index"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save embeddings with original shape and dtype
        torch.save({
            'model_path': self.model_path,
            'embeddings': self.embeddings,
            'max_len': max_len
        }, os.path.join(save_dir, 'clip_embeddings.pt'))
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(save_dir, 'clip_embeddings.faiss'))
            print("Saved CPU index")
    
    @classmethod
    def load(cls, save_dir: str, device: str = "cuda"):
        """Load codebook and index"""
        instance = cls(None, device)
        
        # Load embeddings
        data = torch.load(os.path.join(save_dir, 'clip_embeddings.pt'))
        instance.model_path = data['model_path']
        instance.embeddings = data['embeddings']
        instance.max_len = data.get('max_len', None)
        
        # Load FAISS index
        instance.index = faiss.read_index(os.path.join(save_dir, 'clip_embeddings.faiss'))
        print("Loaded index on CPU")

        return instance
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 5):
        """Search for similar embeddings using cosine similarity"""
        # Flatten query embedding
        query = query_embedding.reshape(1, -1)  # [1, seq_len * hidden_dim]

        # Normalize query vector
        query_norm = np.linalg.norm(query)
        query = query / query_norm

        # Since we normalized both query and index vectors,
        # the inner product will give us cosine similarity
        D, I = self.index.search(
            query.astype(np.float32),
            k
        )

        results = []
        for idx, sim in zip(I[0], D[0]):
            results.append((self.embeddings[idx].prompt, self.embeddings[idx].embedding, sim))

        return results

def load_blank_image(width, height):
    pil_image = Image.new("RGB", (width, height), (255, 255, 255)).convert('RGB')
    return pil_image

def build_inputs(model, text_tokenizer, visual_tokenizer, prompt, pil_image, target_width, target_height):
    if pil_image is not None:
        target_size = (int(target_width), int(target_height))
        pil_image, vae_pixel_values, cond_img_ids = model.visual_generator.process_image_aspectratio(pil_image, target_size)
        cond_img_ids[..., 0] = 1.0
        vae_pixel_values = vae_pixel_values.unsqueeze(0)
        width = pil_image.width
        height = pil_image.height
        resized_height, resized_width = visual_tokenizer.smart_resize(height, width, max_pixels=visual_tokenizer.image_processor.min_pixels)
        pil_image = pil_image.resize((resized_width, resized_height))
    else:
        vae_pixel_values = None
        cond_img_ids = None

    prompt, input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        prompt, 
        [pil_image], 
        generation_preface=None,
        return_labels=False,
        propagate_exception=False,
        multimodal_type='single_image',
        fix_sample_overall_length_navit=False
        )
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    if pixel_values is not None:
        pixel_values = torch.cat([
            pixel_values if pixel_values is not None else None
        ],dim=0)
    if grid_thws is not None:
        grid_thws = torch.cat([
            grid_thws if grid_thws is not None else None
        ],dim=0)
    return input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values

def main():
    parser = argparse.ArgumentParser(description='Analyze CLIP embeddings')
    parser.add_argument('--model_path', type=str, default="runwayml/stable-diffusion-inpainting")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--save_dir', type=str, default="codebook")
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--encoder_idx', type=int, default=None, 
                       help='Encoder index for SD v3.5 (1: CLIP ViT-bigG, 2: CLIP ViT-L, 3: T5)')
    parser.add_argument('--save_all_encoders', action='store_true',
                       help='Save codebooks for all encoders in SD v3.5')
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # Check model types
    is_sd_v3 = "stable-diffusion-3" in args.model_path
    
    # Create and process embeddings for specific encoder(s)
    if is_sd_v3:
        # Determine which encoders to process
        encoders_to_process = []
        if args.save_all_encoders:
            encoders_to_process = [1, 2, 3]
        elif args.encoder_idx is not None:
            encoders_to_process = [args.encoder_idx]
        else:
            # Default to first encoder if nothing specified
            encoders_to_process = [1]
        
        # Process each specified encoder
        for encoder_idx in encoders_to_process:
            print(f"Processing encoder {encoder_idx} for SD v3.5...")
            
            if encoder_idx == 1:
                # CLIP ViT-L
                print("Loading CLIP ViT-L tokenizer and text encoder...")
                tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
                text_encoder = CLIPTextModel.from_pretrained(
                    args.model_path, 
                    subfolder="text_encoder", 
                ).to(device)
                encoder_save_dir = f"{args.save_dir}_encoder1"
            elif encoder_idx == 2:
                # CLIP ViT-bigG
                print("Loading CLIP ViT-bigG tokenizer and text encoder...")
                tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer_2")
                text_encoder = CLIPTextModel.from_pretrained(
                    args.model_path, 
                    subfolder="text_encoder_2", 
                ).to(device)
                encoder_save_dir = f"{args.save_dir}_encoder2"
            elif encoder_idx == 3:
                # T5
                print("Loading T5 tokenizer and text encoder...")
                from transformers import T5Tokenizer, T5EncoderModel
                tokenizer = T5Tokenizer.from_pretrained(args.model_path, subfolder="tokenizer_3")
                text_encoder = T5EncoderModel.from_pretrained(
                    args.model_path,
                    subfolder="text_encoder_3",
                    torch_dtype=torch.float16  # Use float16 instead of bfloat16
                ).to(device)
                encoder_save_dir = f"{args.save_dir}_encoder3"
            
            # Create codebook for this encoder
            codebook = CLIPEmbeddingCodebook(args.model_path, args.device)
            
            # Read prompts from CSV
            df = pd.read_csv(args.csv_path)
            prompts = df['prompt'].tolist()
            
            # Process each prompt with the current encoder
            for prompt in tqdm(prompts, desc=f"Processing encoder {encoder_idx}"):
                tokens = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
                
                # Get embedding - depends on encoder type
                with torch.no_grad():
                    if encoder_idx == 3:  # T5
                        embedding = text_encoder(
                            input_ids=tokens.input_ids,
                            attention_mask=tokens.attention_mask
                        ).last_hidden_state[0]  # [seq_len, hidden_dim]

                        # Keep float16 dtype
                        embedding = embedding.to(torch.float16)
                    else:  # CLIP models
                        embedding = text_encoder(tokens.input_ids)[0][0]  # [seq_len, hidden_dim]
                
                # Add to codebook
                codebook.add_embedding(prompt, embedding)
            
            # Build index and save
            codebook.build_index()
            codebook.save(encoder_save_dir, None)
            print(f"Saved codebook for encoder {encoder_idx} to {encoder_save_dir}")
    else:
        # Original single encoder logic
        tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            args.model_path, 
            subfolder="text_encoder", 
        ).to(device)
        
        # Create CLIP embedding codebook
        codebook = CLIPEmbeddingCodebook(args.model_path, args.device)
        
        # Read prompts from CSV
        df = pd.read_csv(args.csv_path)
        prompts = df['prompt'].tolist()
        
        # Process each prompt
        for prompt in tqdm(prompts):
            tokens = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            # Get CLIP embedding - store full sequence
            with torch.no_grad():
                embedding = text_encoder(tokens.input_ids)[0][0]  # [seq_len, hidden_dim]
            
            # Add to codebook
            codebook.add_embedding(prompt, embedding)
        
        # Build index and save
        codebook.build_index()
        codebook.save(args.save_dir, None)

if __name__ == "__main__":
    main() 