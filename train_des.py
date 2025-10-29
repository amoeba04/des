import torch
from torch import nn
import torch.optim as optim
from transformers import CLIPTokenizer, CLIPTextModel, T5EncoderModel, T5Tokenizer, AutoModelForCausalLM
import argparse
import pandas as pd
import os
import json
from tqdm import tqdm
from dataclasses import dataclass
import numpy as np
from save_codebook import CLIPEmbeddingCodebook, load_blank_image
import faiss

@dataclass
class CLIPEmbeddingInfo:
    prompt: str
    embedding: np.ndarray  # [seq_len, hidden_dim]

def build_inputs(model, text_tokenizer, visual_tokenizer, prompt, pil_image, target_width, target_height):
    if isinstance(prompt, str):
        prompts = [prompt]
    else:
        prompts = list(prompt)

    if isinstance(pil_image, list):
        images = pil_image
        assert len(images) == len(prompts)
    else:
        images = [pil_image] * len(prompts)

    batched_input_ids = []
    batched_attention_masks = []
    pixel_values_list = []
    grid_thws_list = []
    vae_pixel_values_list = []

    for p, img in zip(prompts, images):
        if img is not None:
            target_size = (int(target_width), int(target_height))
            pil_img_proc, vae_pixel_values, _ = model.visual_generator.process_image_aspectratio(img, target_size)
            vae_pixel_values = vae_pixel_values.unsqueeze(0).to(device=model.device)

            width = pil_img_proc.width
            height = pil_img_proc.height
            resized_height, resized_width = visual_tokenizer.smart_resize(height, width, max_pixels=visual_tokenizer.image_processor.min_pixels)
            pil_for_vt = pil_img_proc.resize((resized_width, resized_height))
        else:
            vae_pixel_values = None
            pil_for_vt = None

        _, input_ids, pixel_values, grid_thws = model.preprocess_inputs(
            p,
            [pil_for_vt] if pil_for_vt is not None else [None],
            generation_preface=None,
            return_labels=False,
            propagate_exception=False,
            multimodal_type='single_image',
            fix_sample_overall_length_navit=False
        )

        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        batched_input_ids.append(input_ids)
        batched_attention_masks.append(attention_mask)

        if pixel_values is not None:
            pixel_values_list.append(pixel_values.to(device=visual_tokenizer.device, dtype=torch.bfloat16))
        if grid_thws is not None:
            grid_thws_list.append(grid_thws.to(device=visual_tokenizer.device))
        if vae_pixel_values is not None:
            vae_pixel_values_list.append(vae_pixel_values)

    max_len = max([ids.shape[0] for ids in batched_input_ids])
    padded_input_ids = []
    padded_attention_masks = []
    for ids, mask in zip(batched_input_ids, batched_attention_masks):
        pad_size = max_len - ids.shape[0]
        if pad_size > 0:
            pad_ids = torch.full((pad_size,), text_tokenizer.pad_token_id, dtype=ids.dtype, device=ids.device)
            pad_mask = torch.zeros((pad_size,), dtype=mask.dtype, device=mask.device)
            ids = torch.cat([pad_ids, ids], dim=0)
            mask = torch.cat([pad_mask, mask], dim=0)
        padded_input_ids.append(ids)
        padded_attention_masks.append(mask)

    input_ids_batch = torch.stack(padded_input_ids, dim=0).to(device=model.device)
    attention_mask_batch = torch.stack(padded_attention_masks, dim=0).to(device=model.device)

    pixel_values_batch = None
    grid_thws_batch = None
    vae_pixel_values_batch = None

    if len(pixel_values_list) > 0:
        pixel_values_batch = torch.cat(pixel_values_list, dim=0)
    if len(grid_thws_list) > 0:
        grid_thws_batch = torch.cat(grid_thws_list, dim=0)
    if len(vae_pixel_values_list) > 0:
        vae_pixel_values_batch = torch.cat(vae_pixel_values_list, dim=0)

    return input_ids_batch, pixel_values_batch, attention_mask_batch, grid_thws_batch, vae_pixel_values_batch

def train(args):
    device = torch.device(args.device)

    # Check if using SDv3.5 or Flux
    is_sd_v3 = "stable-diffusion-3" in args.model_path or "flux" in args.model_path.lower()

    # Determine which text encoder to use
    using_text_encoder = 1  # default
    if args.text_encoder_idx is not None:
        using_text_encoder = args.text_encoder_idx

    # Load models based on model type and encoder selection
    if is_sd_v3:
        print(f"Using SD v3.5/Flux multi-encoder model: {args.model_path}")
        if using_text_encoder == 1:
            print("Training text_encoder (CLIP ViT-L)")
            tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(
                args.model_path,
                subfolder="text_encoder"
            ).to(device)
            original_text_encoder = CLIPTextModel.from_pretrained(
                args.model_path,
                subfolder="text_encoder"
            ).to(device)
            embedding_dim = text_encoder.config.hidden_size  # CLIP ViT-L (768)
        elif using_text_encoder == 2:
            print("Training text_encoder_2 (CLIP ViT-bigG)")
            tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer_2")
            text_encoder = CLIPTextModel.from_pretrained(
                args.model_path,
                subfolder="text_encoder_2"
            ).to(device)
            original_text_encoder = CLIPTextModel.from_pretrained(
                args.model_path,
                subfolder="text_encoder_2"
            ).to(device)
            embedding_dim = text_encoder.config.hidden_size  # CLIP ViT-bigG (1280)
        elif using_text_encoder == 3:
            print("Training text_encoder_3 (T5)")
            tokenizer = T5Tokenizer.from_pretrained(args.model_path, subfolder="tokenizer_3")

            # T5 memory optimization options
            t5_config = {
                "torch_dtype": torch.bfloat16,
            }

            # Disable cache for gradient checkpointing
            if args.gradient_checkpointing:
                t5_config["use_cache"] = False

            # Multi-GPU support with device_map="auto"
            if args.multi_gpu:
                print("Using multi-GPU with device_map='auto'")
                t5_config["device_map"] = "auto"
                text_encoder = T5EncoderModel.from_pretrained(
                    args.model_path,
                    subfolder="text_encoder_3",
                    **t5_config
                )
                # Get the first device where model is loaded
                device = next(text_encoder.parameters()).device
            else:
                text_encoder = T5EncoderModel.from_pretrained(
                    args.model_path,
                    subfolder="text_encoder_3",
                    **t5_config
                ).to(device)

            # Load original text encoder (keep on CPU to save memory, move when needed)
            original_text_encoder = T5EncoderModel.from_pretrained(
                args.model_path,
                subfolder="text_encoder_3",
                torch_dtype=torch.bfloat16
            )
            # For multi-GPU, keep original encoder on CPU and move when needed
            if not args.multi_gpu:
                original_text_encoder = original_text_encoder.to(device)

            embedding_dim = text_encoder.config.d_model  # T5 (4096)
    else:
        # Legacy SD v1/v2 models with single text encoder
        print(f"Using SD v1/v2 single-encoder model: {args.model_path}")
        tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            args.model_path,
            subfolder="text_encoder"
        ).to(device)
        original_text_encoder = CLIPTextModel.from_pretrained(
            args.model_path,
            subfolder="text_encoder"
        ).to(device)
        embedding_dim = text_encoder.config.hidden_size  # v1: 768, v2: 1024

    # T5 flag for later use
    is_t5 = is_sd_v3 and using_text_encoder == 3

    # Print device information
    if args.multi_gpu and is_t5:
        print(f"Multi-GPU mode enabled. Model distributed across devices:")
        if hasattr(text_encoder, 'hf_device_map'):
            for name, dev in text_encoder.hf_device_map.items():
                print(f"  {name}: {dev}")
        print(f"Input device: {device}")
    else:
        print(f"Using device: {device}")

    # Get model sequence length
    if is_t5:
        max_length = tokenizer.model_max_length
    else:
        max_length = tokenizer.model_max_length

    # Set text encoder to training mode
    text_encoder.train()

    # Set all parameters to require gradients
    for param in text_encoder.parameters():
        param.requires_grad = True

    # Enable gradient checkpointing for T5 AFTER setting requires_grad
    if is_t5 and args.gradient_checkpointing:
        print("Enabling gradient checkpointing for T5")
        text_encoder.gradient_checkpointing_enable()
        # For T5 with gradient checkpointing, ensure encoder layers have requires_grad
        for name, param in text_encoder.named_parameters():
            param.requires_grad = True
        # Also ensure the shared embeddings have requires_grad
        if hasattr(text_encoder, 'shared'):
            for param in text_encoder.shared.parameters():
                param.requires_grad = True

        # Verify that parameters have requires_grad
        trainable_params = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in text_encoder.parameters())
        print(f"T5 trainable parameters: {trainable_params:,} / {total_params:,}")

    optimizer = optim.AdamW(text_encoder.parameters(), lr=args.learning_rate)
    cos_sim = nn.CosineSimilarity(dim=-1)

    original_text_encoder.eval()

    # Helper function to handle CLIP vs T5 embedding extraction
    def get_embeddings(encoder, tokens, is_t5_model=False, is_original=False):
        """
        Extract embeddings from either CLIP or T5 encoder.
        CLIP returns tuple with embeddings as first element.
        T5 returns object with last_hidden_state attribute.

        Args:
            encoder: The text encoder model
            tokens: Tokenized input
            is_t5_model: Whether this is a T5 model
            is_original: Whether this is the original (non-training) encoder
        """
        if is_t5_model:
            # For multi-GPU original encoder, move to device temporarily
            if is_original and args.multi_gpu:
                # Move original encoder to first GPU temporarily
                first_device = next(text_encoder.parameters()).device
                encoder_temp = encoder.to(first_device)
                with torch.no_grad():
                    result = encoder_temp(tokens.input_ids, attention_mask=tokens.attention_mask).last_hidden_state.to(torch.bfloat16)
                # Move back to CPU to save memory
                encoder.cpu()
                torch.cuda.empty_cache()
                return result
            else:
                # T5 returns last_hidden_state, convert to bfloat16
                return encoder(tokens.input_ids, attention_mask=tokens.attention_mask).last_hidden_state.to(torch.bfloat16)
        else:
            # CLIP returns tuple, first element is text embeddings
            return encoder(tokens.input_ids)[0]

    # Load codebook (with encoder-specific naming for SD v3.5)
    if is_sd_v3:
        codebook_dir = f"{args.codebook_dir}_encoder{using_text_encoder}"
        if not os.path.exists(codebook_dir):
            print(f"Warning: Codebook directory {codebook_dir} does not exist. Using default: {args.codebook_dir}")
            codebook_dir = args.codebook_dir
    else:
        codebook_dir = args.codebook_dir

    # For multi-GPU, load codebook on CPU (will be used for search only)
    codebook_device = 'cpu' if args.multi_gpu else device
    codebook = CLIPEmbeddingCodebook.load(codebook_dir, codebook_device)
    
    # Load prompts
    unsafe_df = pd.read_csv(args.unsafe_csv_path)
    safe_df = pd.read_csv(args.safe_csv_path)
    
    # Apply sampling if ratio is less than 1.0
    if args.sampling_ratio < 1.0:
        print(f"Applying sampling ratio of {args.sampling_ratio}")
        unsafe_df = unsafe_df.sample(frac=args.sampling_ratio, random_state=42)
        safe_df = safe_df.sample(frac=args.sampling_ratio, random_state=42)
        print(f"Sampled {len(unsafe_df)} unsafe prompts and {len(safe_df)} safe prompts.")
    
    unsafe_prompts = unsafe_df['prompt'].tolist()
    safe_prompts = safe_df['prompt'].tolist()
    
    # Get unconditioned and concept embeddings
    empty_prompt = ""
    concept_prompt = args.concept_prompt
    uncond_tokens = tokenizer(
        empty_prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    concept_tokens = tokenizer(
        concept_prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        uncond_embedding = get_embeddings(text_encoder, uncond_tokens, is_t5)
        concept_embedding = get_embeddings(text_encoder, concept_tokens, is_t5)

    # Flatten and normalize concept direction
    seq_len = concept_embedding.shape[1]
    concept_direction = concept_embedding.view(1, -1)  # [1, seq_len*embedding_dim]
    concept_norm = torch.norm(concept_direction, dim=-1, keepdim=True)
    concept_direction = concept_direction / (concept_norm + 1e-12)
    concept_direction = concept_direction[0]  # [seq_len*embedding_dim]

    # Update safe embedding path with encoder suffix for SD v3.5
    if is_sd_v3:
        safe_embedding_path = args.safe_embedding_path
        if safe_embedding_path and not f"_encoder{using_text_encoder}" in safe_embedding_path:
            path_parts = os.path.splitext(safe_embedding_path)
            safe_embedding_path = f"{path_parts[0]}_encoder{using_text_encoder}{path_parts[1]}"
    else:
        safe_embedding_path = args.safe_embedding_path

    # Try to load paired_data if path exists
    if safe_embedding_path is not None and os.path.exists(safe_embedding_path):
        print(f"Loading paired data from {safe_embedding_path}")
        loaded_data = torch.load(safe_embedding_path)
        paired_data = loaded_data['paired_data']
        print("Successfully loaded paired data")
    else:
        print("Computing paired data...")
        # Precompute min and max safe embeddings for unsafe prompts
        precomputed_min_safe_embeddings = []
        precomputed_max_safe_embeddings = []

        for prompt in tqdm(unsafe_prompts, desc='Precomputing safe embeddings'):
            tokens = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                unsafe_embedding = get_embeddings(text_encoder, tokens, is_t5)

            # Flatten for FAISS search
            if is_t5:
                # bfloat16 to float32 for numpy operations
                unsafe_np = unsafe_embedding.detach().float().cpu().numpy()[0]
            else:
                unsafe_np = unsafe_embedding.detach().cpu().numpy()[0]

            unsafe_np_flat = unsafe_np.reshape(1, -1)
            faiss.normalize_L2(unsafe_np_flat)

            # Find all safe embeddings
            codebook_size = len(codebook.embeddings)
            similar_entries = codebook.search_similar(unsafe_np, k=codebook_size)
            
            # Get min safe embedding (most similar)
            min_safe_embedding = torch.from_numpy(similar_entries[0][1])
            if is_t5:
                min_safe_embedding = min_safe_embedding.to(torch.bfloat16)

            # Get max safe embedding (most distant)
            max_safe_embedding = torch.from_numpy(similar_entries[-1][1])
            if is_t5:
                max_safe_embedding = max_safe_embedding.to(torch.bfloat16)

            # Subtract concept direction
            max_safe_flat = max_safe_embedding.view(-1) - args.concept_guidance_scale * concept_direction.cpu()

            try:
                # Reshape to original form
                max_safe_embedding = max_safe_flat.view(seq_len, embedding_dim)
            except:
                # Dimension mismatch handling
                print(f"Warning: Dimension mismatch! Concept: {concept_direction.shape}, Max safe: {max_safe_flat.shape}")
                max_safe_embedding = max_safe_flat.view(seq_len, -1)[:, :embedding_dim]

            precomputed_min_safe_embeddings.append(min_safe_embedding.unsqueeze(0))
            precomputed_max_safe_embeddings.append(max_safe_embedding.unsqueeze(0))

        # Create paired datasets
        if len(unsafe_prompts) > len(safe_prompts):
            safe_prompts_extended = safe_prompts * (len(unsafe_prompts) // len(safe_prompts)) + \
                                  safe_prompts[:len(unsafe_prompts) % len(safe_prompts)]
            paired_data = list(zip(
                unsafe_prompts,
                precomputed_min_safe_embeddings,
                precomputed_max_safe_embeddings,
                safe_prompts_extended
            ))
        else:
            unsafe_prompts_extended = unsafe_prompts * (len(safe_prompts) // len(unsafe_prompts)) + \
                                    unsafe_prompts[:len(safe_prompts) % len(unsafe_prompts)]
            min_safe_embeddings_extended = precomputed_min_safe_embeddings * (len(safe_prompts) // len(unsafe_prompts)) + \
                                         precomputed_min_safe_embeddings[:len(safe_prompts) % len(unsafe_prompts)]
            max_safe_embeddings_extended = precomputed_max_safe_embeddings * (len(safe_prompts) // len(unsafe_prompts)) + \
                                         precomputed_max_safe_embeddings[:len(safe_prompts) % len(unsafe_prompts)]
            paired_data = list(zip(
                unsafe_prompts_extended,
                min_safe_embeddings_extended,
                max_safe_embeddings_extended,
                safe_prompts
            ))

        # Save paired data if path is provided
        if safe_embedding_path is not None:
            print(f"Saving paired data to {safe_embedding_path}")
            save_dir = os.path.dirname(safe_embedding_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'paired_data': paired_data,
                'args': vars(args),  # for reproducibility
                'using_text_encoder': using_text_encoder,  # which encoder was used
                'is_t5': is_t5  # whether T5 encoder was used
            }, safe_embedding_path)
            print("Successfully saved paired data")
    
    # Training loop
    for epoch in range(args.num_epochs):
        total_loss = 0
        total_unsafe_loss = 0
        total_safe_loss = 0
        
        # Shuffle pairs for each epoch
        np.random.shuffle(paired_data)
        
        # Create batches
        for i in tqdm(range(0, len(paired_data), args.batch_size), desc=f'Epoch {epoch+1}/{args.num_epochs}'):
            # Unpack batch
            batch = paired_data[i:i + args.batch_size]
            batch_unsafe_prompts, batch_min_safe_embeddings, batch_max_safe_embeddings, batch_safe_prompts = zip(*batch)
            
            optimizer.zero_grad()
            
            # Process unsafe/safe prompts
            unsafe_tokens = tokenizer(
                list(batch_unsafe_prompts),
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            safe_tokens = tokenizer(
                list(batch_safe_prompts),
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            # Stack embeddings and move to device
            if is_t5:
                batch_max_safe_embeddings = torch.stack([emb.to(device) for emb in batch_max_safe_embeddings])
            else:
                batch_max_safe_embeddings = torch.stack(batch_max_safe_embeddings).to(device)

            # Get current embeddings
            unsafe_current_embedding = get_embeddings(text_encoder, unsafe_tokens, is_t5)
            safe_current_embedding = get_embeddings(text_encoder, safe_tokens, is_t5)
            
            # Calculate losses
            total_step_loss = 0
            total_unsafe_step_loss = 0
            total_safe_step_loss = 0
            
            # 1. safe preservation losses
            if 1 in args.ablation:
                with torch.no_grad():
                    original_safe_outputs = get_embeddings(original_text_encoder, safe_tokens, is_t5, is_original=True)

                safe_loss = 1 - cos_sim(
                    safe_current_embedding.reshape(safe_current_embedding.size(0), -1),
                    original_safe_outputs.reshape(original_safe_outputs.size(0), -1)
                ).mean()
            
                # Concept + Safe -> Safe loss (ConceptAug)
                safe_plus_concept = safe_current_embedding.reshape(safe_current_embedding.size(0), -1) + \
                                  args.concept_guidance_scale * concept_direction.unsqueeze(0)
                safe_concept_sum_loss = 1 - cos_sim(
                    safe_plus_concept,
                    original_safe_outputs.reshape(original_safe_outputs.size(0), -1)
                ).mean()
                total_step_loss += args.lambda_safe * (safe_loss + safe_concept_sum_loss)
                total_safe_step_loss += args.lambda_safe * (safe_loss + safe_concept_sum_loss)
            
            # 2. unsafe -> (max safe - concept direction) loss
            # (max safe - concept direction) pre-computed
            if 2 in args.ablation:
                unsafe_loss = 1 - cos_sim(
                    unsafe_current_embedding.reshape(unsafe_current_embedding.size(0), -1),
                    batch_max_safe_embeddings.reshape(batch_max_safe_embeddings.size(0), -1)
                ).mean()
                total_step_loss += (1 - args.lambda_safe) * unsafe_loss
                total_unsafe_step_loss += (1 - args.lambda_safe) * unsafe_loss
            
            # Get current concept embedding
            current_concept_embedding = get_embeddings(text_encoder, concept_tokens, is_t5)

            # 3. concept to uncond loss
            if 3 in args.ablation:
                concept_to_uncond_loss = 1 - cos_sim(
                    current_concept_embedding.reshape(current_concept_embedding.size(0), -1),
                    uncond_embedding.reshape(uncond_embedding.size(0), -1)
                ).mean()
                total_step_loss += (1 - args.lambda_safe) * concept_to_uncond_loss
                total_unsafe_step_loss += (1 - args.lambda_safe) * concept_to_uncond_loss
            
            # Combine all losses
            total_step_loss.backward()
            optimizer.step()

            # Accumulate losses
            total_loss += total_step_loss.item() * len(batch)
            try:
                total_unsafe_loss += total_unsafe_step_loss.item() * len(batch)
                total_safe_loss += total_safe_step_loss.item() * len(batch)
            except:
                total_unsafe_loss += total_unsafe_step_loss * len(batch)
                total_safe_loss += total_safe_step_loss * len(batch)
        
        # Calculate average losses
        avg_loss = total_loss / len(paired_data)
        avg_unsafe_loss = total_unsafe_loss / len(paired_data)
        avg_safe_loss = total_safe_loss / len(paired_data)
        
        print(f'Epoch {epoch+1}:')
        print(f'  Average Total Loss: {avg_loss:.4f}')
        print(f'  Average Unsafe Step Loss: {avg_unsafe_loss:.4f}')
        print(f'  Average Safe Step Loss: {avg_safe_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            # Create directory for the specific training method and learning rate
            # Include encoder info for SD v3.5
            if is_sd_v3:
                method_lr_dir = os.path.join(args.output_dir,
                                           f"encoder{using_text_encoder}_{args.learning_rate}_{args.lambda_safe}")
            else:
                method_lr_dir = os.path.join(args.output_dir, f"{args.learning_rate}_{args.lambda_safe}")

            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(method_lr_dir, exist_ok=True)

            checkpoint = {
                'model_state_dict': text_encoder.state_dict(),
                'epoch': epoch + 1,
                'using_text_encoder': using_text_encoder,
                'is_t5': is_t5,
                'is_sd_v3': is_sd_v3,
                'embedding_dim': embedding_dim,
                'args': vars(args)
            }
            torch.save(
                checkpoint,
                os.path.join(method_lr_dir, f'checkpoint-{epoch+1}.pt')
            )

            # Save metadata separately for easier loading
            metadata = {
                'using_text_encoder': using_text_encoder,
                'is_t5': is_t5,
                'is_sd_v3': is_sd_v3,
                'model_type': 'T5EncoderModel' if is_t5 else 'CLIPTextModel',
                'subfolder': f"text_encoder_{using_text_encoder}" if using_text_encoder > 1 else "text_encoder",
                'embedding_dim': embedding_dim,
                'args': vars(args)
            }
            metadata_path = os.path.join(method_lr_dir, f'checkpoint-{epoch+1}-metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved checkpoint and metadata to {method_lr_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="runwayml/stable-diffusion-inpainting")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--codebook_dir', type=str, default="codebook")
    parser.add_argument('--unsafe_csv_path', type=str, required=True)
    parser.add_argument('--safe_csv_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="checkpoints")
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--sampling_ratio', type=float, default=1.0,
                       help='Ratio of data to sample from each CSV (e.g., 0.2 for 20%)')
    
    # Training arguments
    parser.add_argument('--lambda_safe', type=float, default=0.5,
                       help='Weight for safe loss')
    parser.add_argument('--concept_prompt', type=str, required=True,
                       help='Concept prompt to determine direction')
    parser.add_argument('--concept_guidance_scale', type=float, default=100.0,
                       help='Guidance scale for concept direction')
    
    # etc.
    parser.add_argument('--safe_embedding_path', type=str, default=None,
                       help='Path to safe embedding')
    
    # ablation
    parser.add_argument('--ablation', nargs='+', type=int, default=[1, 2, 3],
                       help='List of losses to ablate (1: unsafe->(max safe - concept direction), '
                            '2: safe preserve with conceptaug, '
                            '3: concept->uncond)')

    # Multi-encoder support (SD v3.5, Flux)
    parser.add_argument('--text_encoder_idx', type=int, default=None, choices=[1, 2, 3],
                       help='Which text encoder to train for SD v3.5/Flux (1: CLIP ViT-L, 2: CLIP ViT-bigG, 3: T5)')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Enable gradient checkpointing to save memory (useful for T5)')
    parser.add_argument('--multi_gpu', action='store_true',
                       help='Use multi-GPU with device_map="auto" for large models like T5')

    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()